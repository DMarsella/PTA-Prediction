#!/usr/bin/env python
from __future__ import division
import matplotlib
import matplotlib.pyplot as plt

import numpy as np, pickle
import math, sys, os, glob, h5py, json
from astropy import units as u
from sklearn.neighbors import KernelDensity

import pint
from pint import toa
from pint import models
from pint.residuals import Residuals
from pint.simulation import make_fake_toas_fromMJDs
pint.logging.setup(sink=sys.stderr, level="WARNING", usecolors=True)

import pta_replicator
from pta_replicator import simulate
from pta_replicator import white_noise
#from h5pulsar.pulsar import Pulsar, FilePulsar

'''
def hdf5_pop(filepath, num):

    """
    Function that returns a list of pulsars from a given data set of hdf5 files, sorted by observation time.

    :param filepath: the location of the hdf5 files.
    :param num: Number of pulsars desired in the list.

    :return psrs: a list of pulsar objects.
    """

    print('Filepath is ' + filepath)
    print(f'Number of Pulsars requested is {num}')
    
    pdur = []
    temp = []
    psrs = []
    
    #populate a list of pulsars
    for hdf_file in glob.glob(filepath + '*.hdf5'):        
        psr = FilePulsar(hdf_file)
        temp.append(psr)
    print('Loaded {0} pulsars'.format(len(temp)))

    #determine observation times of each pulsar
    for i in range(len(temp)):
        obtime = max(temp[i].toas) - min(temp[i].toas)
        pdur.append(obtime)
        #print(str(i) + ': ' + str(obtime))

    if num > len(temp):
        num = len(temp)
        
    #Find Maximum 'num' entries in pdur and apply them to the psrs list
    for j in range(num):
        index = pdur.index(max(pdur))
        #print(str(index) + '--' + str(pdur[index]))
        psrs.append(temp[index])
        pdur.pop(index)
        temp.pop(index)
        #print(pdur)
    print('Returned {0} pulsars'.format(len(psrs)))
        
    return psrs
'''

def make_residual_plot(psr, save=False, simdir='pint_sims1/'):
    
    fig, axs = plt.subplots(1, 1)

    flags = list(np.unique(psr.toas['f']))

    for f in flags:

        idx = (psr.toas['f'] == f)
    
        axs.errorbar(psr.toas[idx].get_mjds(), psr.residuals.calc_time_resids()[idx], marker='.', ls='', alpha=0.5, 
                     yerr=psr.toas[idx].get_errors(), label=f)

    plt.title('{0} -- {1} TOAs'.format(psr.name, len(psr.toas)))
    plt.legend()
    plt.tight_layout();

    if save:
        plt.savefig(simdir + '/{0}.png'.format(psr.name))

def create_gw_antenna_pattern(pos, gwtheta, gwphi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).
    :param pos: Unit vector from Earth to pulsar
    :param gwtheta: GW polar angle in radians
    :param gwphi: GW azimuthal angle in radians

    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the
             pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = np.array([np.sin(gwphi), -np.cos(gwphi), 0.0])
    n = np.array([-np.cos(gwtheta) * np.cos(gwphi), -np.cos(gwtheta) * np.sin(gwphi), np.sin(gwtheta)])
    omhat = np.array([-np.sin(gwtheta) * np.cos(gwphi), -np.sin(gwtheta) * np.sin(gwphi), -np.cos(gwtheta)])

    fplus = 0.5 * (np.dot(m, pos) ** 2 - np.dot(n, pos) ** 2) / (1 + np.dot(omhat, pos))
    fcross = (np.dot(m, pos) * np.dot(n, pos)) / (1 + np.dot(omhat, pos))
    cosMu = -np.dot(omhat, pos)

    return fplus, fcross, cosMu

def gw_contribution(chirp, lumin_d, gw_freq, phi_0, pulsar, cosi):
    """
    Function to generate the s cross and s plus contributions to the GW affects on a residual,
    defined in Ellis, Siemens, and Creighton (2012).

    :param chirp: Chirp mass of source binary, received as billions of Solar Masses
    :param lumin_d: Luminosity distance to source, received in Megaparsecs
    :param gw_freq: gravitational wave frequency, received in nHz
    :param phi_0: Initial Phase of the binary in radians
    :param pulsar: a pulsar object
    :param cosi: cos of inclination angle of source

    :return: (splus, scross), where splus and scross are the gw contribution (in seconds) to the sinosoid s function defined in Ellis, Siemens, and Creighton.
    """

    #generate pulsar distance in kpc.
    pdist = 1/psr.model.PX.value
    #print(pdist)
    #print(type(pdist))
    
    #convert units to seconds
    M_c = chirp * 5e-6 * 1e9
    d_L = lumin_d * (3.086e22/3e8)
    f_GW = gw_freq * 1e-9
    etime = pulsar.toas.get_mjds().to(u.s).value
    #print(etime)
    ptime = etime - (pdist * 3000 * 3e7)
    
    #Generate terms
    omega_0 = np.pi * f_GW
    a = (M_c**(5/3) * (1 + cosi**2)) / d_L
    b = (M_c**(5/3) * cosi) / d_L

    e_omega = (omega_0**(-8/3) - (256/5) * (M_c**(5/3)) * etime)**(-3/8)
    p_omega = (omega_0**(-8/3) - (256/5) * (M_c**(5/3)) * ptime)**(-3/8)

    e_phi = phi_0 + (1/32) * M_c**(-5/3) * (omega_0**(-5/3) - e_omega**(-5/3))
    p_phi = phi_0 + (1/32) * M_c**(-5/3) * (omega_0**(-5/3) - p_omega**(-5/3))
    
    #generate s-plus and s-cross.
    splus = a * ((-np.sin(2*e_phi) / e_omega**(1/3)) - (-np.sin(2 * p_phi) / p_omega**(1/3)))
    scross = b * ((2 * np.cos(2*e_phi) / e_omega**(1/3)) - (2 * np.cos(2*e_phi) / p_omega**(1/3)))

    #print(etime)

    return splus, scross

def make_residual_plot(psr, save=False, simdir='pint_sims1/'):
    
    fig, axs = plt.subplots(1, 1)

    flags = list(np.unique(psr.toas['f']))

    for f in flags:

        idx = (psr.toas['f'] == f)
    
        axs.errorbar(psr.toas[idx].get_mjds(), psr.residuals.calc_time_resids()[idx], marker='.', ls='', alpha=0.5, 
                     yerr=psr.toas[idx].get_errors(), label=f)

    plt.title('{0} -- {1} TOAs'.format(psr.name, len(psr.toas)))
    plt.legend()
    plt.tight_layout();

    if save:
        plt.savefig(simdir + '/{0}.png'.format(psr.name))

def compute_daily_ave(times, res, err, ecorr=None, dt=1.0, flags=None):
    """
    From PAL2
    ...
    Computes daily averaged residuals 
     :param times: TOAs in seconds
     :param res: Residuals in seconds
     :param err: Scaled (by EFAC and EQUAD) error bars in seconds
     :param ecorr: (optional) ECORR value for each point in s^2 [default None]
     :param dt: (optional) Time bins for averaging [default 1 s]
     :return: Average TOAs in seconds
     :return: Average error bars in seconds
     :return: Average residuals in seconds
     :return: (optional) Average flags
     """

    isort = np.argsort(times)

    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])

    avetoas = np.array([np.mean(times[l]) for l in bucket_ind],'d')

    if flags is not None:
        aveflags = np.array([flags[l[0]] for l in bucket_ind])
    
    aveerr = np.zeros(len(bucket_ind))
    averes = np.zeros(len(bucket_ind))

    for i,l in enumerate(bucket_ind):
        M = np.ones(len(l))
        C = np.diag(err[l]**2) 
        if ecorr is not None:
            C += np.ones((len(l), len(l))) * ecorr[l[0]]

        avr = 1/np.dot(M, np.dot(np.linalg.inv(C), M))
        aveerr[i] = np.sqrt(avr)
        averes[i] = avr * np.dot(M, np.dot(np.linalg.inv(C), res[l]))

    if flags is not None:
        return avetoas, aveerr, averes, aveflags
    else:
        return avetoas, aveerr, averes

def make_new_toas(newobs, old_toas):
    
    idx = (old_toas['f'] == newobs['f']) #old_toas edited from avg_toas
    log10_errs = np.log10(old_toas[idx].get_errors().to(u.s).value)

    mykde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(log10_errs.reshape((log10_errs.size, 1)))
    
    #samples = mykde.sample(100000)
    #plt.hist(log10_errs, histtype='step', bins=50, density=True);
    #plt.hist(samples, histtype='step', bins=50, density=True);

    delta_t = 365.24/newobs['cadence']
    #print(delta_t)

    if newobs['start'] is None:
        start = max(old_toas[idx].get_mjds()).value + delta_t
    else:
        start = newobs['start']

    #print(start, newobs['end'])
    mjds = np.arange(start, newobs['end'], delta_t)
    #print(len(mjds))

    obs = old_toas[idx][0]['obs'].value[0]
    errs = 10**mykde.sample(len(mjds)).reshape(len(mjds))
    #plt.hist(np.log10(errs), density=True)
    #plt.hist(log10_errs, histtype='step', bins=50, density=True);

    return toa.get_TOAs_array(mjds, obs=obs, flags={'f': newobs['f']}, 
                              errors=errs*1e6, planets=True, ephem='DE440')

def generate_daily_avg_toas(psr, secperday):

    simulate.make_ideal(psr)
    
    flags = list(np.unique(psr.toas['f']))
    idx = (psr.toas['f'] == flags[0])

    print('Pulsar {0} has {1} TOAs observed with {2} systems...'.format(psr.name, len(idx), len(flags)))

    mytoas = psr.toas[idx]
    myresiduals = np.zeros(len(psr.toas[idx]))
    print('Filtering out {0} TOAs with flag {1} observed with {2}...'.format(len(np.where(idx == True)[0]), 
                                                                             flags[0], mytoas['obs'][0]))
    
    # get scaled errors
    err = psr.model.scale_toa_sigma(mytoas).to(u.s).value

    # get ecorr
    U, ecorrvec = psr.model.ecorr_basis_weight_pair(mytoas)
    ecorr = np.dot(U*ecorrvec, np.ones(U.shape[1]))

    avetoas, aveerr, averes = compute_daily_ave(mytoas.get_mjds().to(u.s).value, 
                                                myresiduals, err, ecorr=ecorr, dt=secperday)

    toas2 = toa.get_TOAs_array(avetoas/secperday, obs=mytoas['obs'][0], flags={'f': flags[0]}, errors=aveerr*1e6, 
                               planets=True, ephem='DE440')
    
    for f in flags[1:]:
        idx = (psr.toas['f'] == f)
        mytoas = psr.toas[idx]
        print('Filtering out {0} TOAs with flag {1} observed with {2}...'.format(len(np.where(idx == True)[0]), f, 
                                                                                 mytoas['obs'][0]))
        myresiduals = np.zeros(len(psr.toas[idx]))
    
        # get scaled errors
        err = psr.model.scale_toa_sigma(mytoas).to(u.s).value

        # get ecorr
        U, ecorrvec = psr.model.ecorr_basis_weight_pair(mytoas)
        ecorr = np.dot(U*ecorrvec, np.ones(U.shape[1]))

        avetoas, aveerr, averes = compute_daily_ave(mytoas.get_mjds().to(u.s).value, 
                                                    myresiduals, err, ecorr=ecorr, dt=secperday)

        toas2.merge(toa.get_TOAs_array(avetoas/secperday, obs=mytoas['obs'][0], flags={'f': f}, errors=aveerr*1e6, 
                                       planets=True, ephem='DE440'))

    return toas2

def DailyAvgPop(datadir, timeline):
    secperday = 24*3600

    old_parfiles = sorted(glob.glob(datadir + 'par/*.par'))
    old_timfiles = sorted(glob.glob(datadir + 'tim/*.tim'))
    print(f'Loaded par({len(old_parfiles)}) and tim({len(old_timfiles)}) files')

    psrs = []
    
    for i in range(len(old_parfiles)):
        psr = simulate.load_pulsar(old_parfiles[i], old_timfiles[i], ephem='DE440')

        if 'PLRedNoise' in psr.model.components.keys():
            psr.model.remove_component('PLRedNoise')

        avg_toas = generate_daily_avg_toas(psr, secperday)
        newtoas = []
        newobslist = []

        for f in list(np.unique(psr.toas['f'])):
            newobslist.append({'f': f, 'start': None, 'end': timeline, 'cadence': 20})

        print(newobslist)
            
        for newobs in newobslist:
            newtoas.append(make_new_toas(newobs, avg_toas))

        for toas in newtoas:
            avg_toas.merge(toas)

        psr.toas = avg_toas
        print('Pulsar {0} now has {1} daily averaged TOAs'.format(psr.name, len(psr.toas)))

        # remove EcorrNoise and ScaleToaError from the timing model
        print('Removing measurement noise and ECORR...')
        psr.model.remove_component('EcorrNoise')
        psr.model.remove_component('ScaleToaError')
        
        # go through and remove any maskParameters that are now empty
        empty_masks = psr.model.find_empty_masks(psr.toas)
        if len(empty_masks) > 0:
            for m in empty_masks:
                psr.model.remove_param(m)
        
        # get a list of the model components
        component_names = psr.model.components.keys()
            
        # remove any components that no longer have any parameters
        for name in component_names:
            if len(psr.model.components[name].params) == 0:
                psr.model.remove_component(name)
        
        # remove DMX and troposphere delay
        if 'DispersionDMX' in component_names:
            psr.model.remove_component('DispersionDMX')
        if 'TroposphereDelay' in component_names:
            psr.model.remove_component('TroposphereDelay')
        if 'FD' in component_names:
            psr.model.remove_component('FD')
                
        simulate.make_ideal(psr)
        white_noise.add_measurement_noise(psr, efac=1)
        for _ in range(3):
            psr.fit()

        psrs.append(psr)

    return psrs


        


