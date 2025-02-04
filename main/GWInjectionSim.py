#!/usr/bin/env python

from enterprise.signals import signal_base
import glob
from decimal import Decimal
from memory_profiler import profile

from enterprise_extensions.deterministic import cw_delay, CWSignal
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import scipy.sparse as ss
from sksparse.cholmod import cholesky
from enterprise.signals import signal_base
from enterprise.signals.gp_signals import get_timing_model_basis, BasisGP
from enterprise.signals.parameter import function
import pickle
import enterprise.signals.parameter as parameter
from enterprise.signals import utils, signal_base, selections, white_signals, gp_signals, deterministic_signals
from enterprise_extensions import model_utils, blocks, models
from enterprise.signals import gp_priors as gpp
import os
import enterprise
from enterprise_extensions.sampler import get_parameter_groups
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import json
from pathlib import Path
import utility

@profile
def simulate(pta, params, sparse_cholesky=True):
    """Simulate code with enterprise (instead of libstempo/PINT)"""
    delays, ndiags, fmats, phis = (pta.get_delay(params=params),
                                   pta.get_ndiag(params=params),
                                   pta.get_basis(params=params),
                                   pta.get_phi(params=params))

    gpresiduals = []
    if pta._commonsignals:
        if sparse_cholesky:
            cf = cholesky(ss.csc_matrix(phis))
            gp = np.zeros(phis.shape[0])
            gp[cf.P()] = np.dot(cf.L().toarray(), np.random.randn(phis.shape[0]))
        else:
            gp = np.dot(sl.cholesky(phis, lower=True), np.random.randn(phis.shape[0]))

        i = 0
        for fmat in fmats:
            j = i + fmat.shape[1]
            gpresiduals.append(np.dot(fmat, gp[i:j]))
            i = j

        assert len(gp) == i
    else:
        for fmat, phi in zip(fmats, phis):
            if phi is None:
                gpresiduals.append(0)
            elif phi.ndim == 1:
                gpresiduals.append(np.dot(fmat, np.sqrt(phi) * np.random.randn(phi.shape[0])))
            else:
                raise NotImplementedError

    whiteresiduals = []
    for delay, ndiag in zip(delays, ndiags):
        if ndiag is None:
            whiteresiduals.append(0)
        elif isinstance(ndiag, signal_base.ShermanMorrison):
            # this code is very slow...
            n = np.diag(ndiag._nvec)
            for j,s in zip(ndiag._jvec, ndiag._slices):
                n[s,s] += j
            whiteresiduals.append(delay + np.dot(sl.cholesky(n, lower=True), np.random.randn(n.shape[0])))
        elif ndiag.ndim == 1:
            whiteresiduals.append(delay + np.sqrt(ndiag) * np.random.randn(ndiag.shape[0]))
        else:
            raise NotImplementedError

    return [np.array(g + w) for g, w in zip(gpresiduals, whiteresiduals)]

@profile
def set_residuals(psr, y):
    psr._residuals[psr._isort] = y

@profile
@function
def tm_prior(weights, toas, variance=1e-14):
    return weights * variance * len(toas)

@profile
def TimingModel(coefficients=False, name="linear_timing_model",
                use_svd=False, normed=True, prior_variance=1e-14):
    """Class factory for marginalized linear timing model signals."""

    basis = get_timing_model_basis(use_svd, normed)
    prior = tm_prior(variance=prior_variance)

    BaseClass = BasisGP(prior, basis, coefficients=coefficients, name=name)

    class TimingModel(BaseClass):
        signal_type = "basis"
        signal_name = "linear timing model"
        signal_id = name + "_svd" if use_svd else name

    return TimingModel

@profile
def cw_block_circ(amp_prior='log-uniform', dist_prior=None,
                  skyloc=None, log10_fgw=None,
                  psrTerm=False, tref=57387*86400, name='cw'):
    
    """
    Returns deterministic, cirular orbit continuous GW model:

    :param amp_prior:
        Prior on log10_h. Default is "log-uniform."
        Use "uniform" for upper limits, or "None" to search over
        log10_dist instead.
    :param dist_prior:
        Prior on log10_dist. Default is "None," meaning that the
        search is over log10_h instead of log10_dist. Use "log-uniform"
        to search over log10_h with a log-uniform prior.
    :param skyloc:
        Fixed sky location of CW signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param log10_fgw:
        Fixed log10 GW frequency of CW signal search.
        Search over GW frequency if ``None`` given.
    :param ecc:
        Fixed log10 distance to SMBHB search.
        Search over distance or strain if ``None`` given.
    :param psrTerm:
        Boolean for whether to include the pulsar term. Default is False.
    :param name:
        Name of CW signal.

    """

    if dist_prior is None:
        log10_dist = None

        if amp_prior == 'uniform':
            log10_h = parameter.LinearExp(-18.0, -11.0)('{}_log10_h'.format(name))
        elif amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(-18.0, -11.0)('{}_log10_h'.format(name))

    elif dist_prior == 'log-uniform':
        log10_dist = parameter.Uniform(-2.0, 4.0)('{}_log10_dL'.format(name))
        log10_h = None

    # chirp mass [Msol]
    log10_Mc = parameter.Uniform(6.0, 10.0)('{}_log10_Mc'.format(name))

    # GW frequency [Hz]
    if log10_fgw is None:
        log10_fgw = parameter.Uniform(-9.0, -7.0)('{}_log10_fgw'.format(name))
    else:
        log10_fgw = parameter.Constant(log10_fgw)('{}_log10_fgw'.format(name))
    # orbital inclination angle [radians]
    cosinc = parameter.Uniform(-1.0, 1.0)('{}_cosinc'.format(name))
    # initial GW phase [radians]
    phase0 = parameter.Uniform(0.0, 2*np.pi)('{}_phase0'.format(name))

    # polarization
    psi_name = '{}_psi'.format(name)
    psi = parameter.Uniform(0, np.pi)(psi_name)

    # sky location
    costh_name = '{}_costheta'.format(name)
    phi_name = '{}_phi'.format(name)
    if skyloc is None:
        costh = parameter.Uniform(-1, 1)(costh_name)
        phi = parameter.Uniform(0, 2*np.pi)(phi_name)
    else:
        costh = parameter.Constant(skyloc[0])(costh_name)
        phi = parameter.Constant(skyloc[1])(phi_name)

    if psrTerm:
        # orbital phase
        p_phase = parameter.Uniform(0, np.pi)
        p_dist = parameter.Normal(0, 1)
    else:
        p_phase = None
        p_dist = 0

    # continuous wave signal
    wf = cw_delay(cos_gwtheta=costh, gwphi=phi, cos_inc=cosinc,
                  log10_mc=log10_Mc, log10_fgw=log10_fgw,
                  log10_h=log10_h, log10_dist=log10_dist,
                  phase0=phase0, psi=psi,
                  psrTerm=True, p_dist=p_dist, p_phase=p_phase,
                  phase_approx=True, check=False,
                  tref=tref)
    cw = CWSignal(wf, ecc=False, psrTerm=psrTerm)

    return cw

@profile
def generate(psrs, rn_dict, CW):
    Tspan = model_utils.get_tspan(psrs)
    s = TimingModel(use_svd=True)

    #noise model
    efac = parameter.Constant(1.0)
    s += white_signals.MeasurementNoise(efac=efac)
    s += blocks.red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan, components=30)
    s += cw_block_circ()

    model = [s(psr) for psr in psrs]
    pta = signal_base.PTA(model)
    print(f'PTA ({type(pta)}) loaded')

    npars = len(pta.params)
    print(f'Loaded {npars} parameters')

    rn_dict.update({"cw_cosinc": CW[0], "cw_psi": CW[7], "cw_phase0": CW[5], "cw_log10_h": CW[4], 
                "cw_costheta": CW[1], "cw_phi": CW[6], 
                "cw_log10_Mc": CW[2], "cw_log10_fgw": CW[3]})
    
    sims = simulate(pta, rn_dict)
    print(f'{len(sims)} simulations loaded')
    print(str(type(sims[0])))

    for i,psr in enumerate(psrs):
        set_residuals(psr, sims[i])

    return psrs


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--pnumber', help='How many pulsars to simulate? Will select for longest observation period from data.', type=int, default=48)
    parser.add_argument('-r', '--repetitions', help='How many simulations to run? Defaults to 1.', type=int, default=1)
    parser.add_argument('-s', '--save', help='Where do you want the data? Defaults to CWD/Data/Sims', default=(str(Path.cwd()) + '/Data/Sims/'))
    parser.add_argument('-d', '--data', help='Where is the data located? Defaults to CWD/Data', default=(str(Path.cwd()) + '/Data/'))
    parser.add_argument('-i', '--inclination', help='Source Inclination in radians.', type=float, default=0)
    parser.add_argument('-mc', '--chirpmass', help='log of Chirp Mass in solar masses', type=float, default=9)
    parser.add_argument('--phase0', help='Initial phase', type=float, default=0)
    parser.add_argument('--psi', help='SOurce Orientation variable', type=float, default=0)
    parser.add_argument('--hmin', help='The lowest h value (log) to simulate.', type = float, default=-15)
    parser.add_argument('--hmax', help='The highest h value (log) to simulate.', type=float, default=-13)
    parser.add_argument('--hnum', help='Increments the h value range into hnum increments.', type=int, default=1)
    parser.add_argument('--rasmin', help='The low point in the range for the Right Ascension, in hours.', type=float, default=12)
    parser.add_argument('--rasmax', help='The high point in the range for the Right Ascension, in hours.', type=float, default=24)
    parser.add_argument('--rasnum', help='Increments the Right Ascension range into rasnum increments.', type=int, default=1)
    parser.add_argument('--decmin', help='Minimum for the Declination range, in degrees.', type=float, default=-15)
    parser.add_argument('--decmax', help='Maximum for the Declination range, in degrees.', type=float, default=15)
    parser.add_argument('--decnum', help='Increments the Declination range into decnum increments.', type=int, default=1)
    parser.add_argument('--fgwmin', help='Minimum for the GW Frequency range, in Hz.', type=float, default=2e-9)
    parser.add_argument('--fgwmax', help='Maximum for the GW Frequency range, in Hz.', type=float, default=1e-7)
    parser.add_argument('--fgwnum', help='Increments the Frequency range into fgwnum increments.', type=int, default=1)

    args = parser.parse_args()

    #Basic Parameters and file information
    save = args.save
    data = args.data
    repetitions = args.repetitions
    pnum = args.pnumber

    #Continuous Wave Parameters
    cos_i = np.cos(args.inclination)
    Mc = args.chirpmass
    phase0 = args.phase0
    psi = args.psi
    hmin=args.hmin
    hmax=args.hmax
    hnum=args.hnum
    rasmin=args.rasmin
    rasmax=args.rasmax
    rasnum=args.rasnum
    decmin=args.decmin
    decmax=args.decmax
    decnum=args.decnum
    fgwmin=args.fgwmin
    fgwmax=args.fgwmax
    fgwnum=args.fgwnum

    #create arrays for the different values to simulate
    harray = np.linspace(hmin, hmax, hnum, dtype=float)
    rasarray = np.linspace(rasmin, rasmax, rasnum, dtype=float)
    decarray = np.linspace(decmin, decmax, decnum, dtype=float)
    fgwarray = np.linspace(np.log10(fgwmin), np.log10(fgwmax), fgwnum, dtype=float)

    #Red Noise Dictionary
    with open(data + 'RedNoiseLibrary.json', 'r') as openfile:
        rn_dict = json.load(openfile)
        
    #Populate the list of pulsars
    psrs = utility.hdf5_pop(data + 'hdf5/', pnum)

    #Generate a range of simulations and pickle the results
    for i in np.nditer(harray):
        hvalue = i*1
    
        for j in np.nditer(rasarray):
            phi = j/24*360
            r = j*1
            
            for k in np.nditer(decarray):
                cos_theta = np.cos(np.pi * (90-k) / 180)
                d = k*1
                
                for l in np.nditer(fgwarray):
                    fgw = l*1
                    fnhz = 10**(l+9)
                    
                    for m in range(repetitions):
                        CW = [cos_i, cos_theta, Mc, fgw, hvalue, phase0, phi, psi]
                        
                        simulations = generate(psrs, rn_dict, CW)                    
                        filename = f'R{Decimal(r):.4}_D{Decimal(d):.4}_h{Decimal(hvalue):.4}_f{Decimal(fnhz):.4}_{pnum}_{m}.pkl'
                        print(filename)
                        
                        with open(save + filename, 'wb') as f:
                            pickle.dump(simulations, f)
