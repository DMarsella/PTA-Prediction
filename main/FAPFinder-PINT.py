#!/usr/bin/env python

import GWInjectionSim

import numpy as np
from pathlib import Path
import glob, math, time, json, pickle, utility
from decimal import Decimal
from memory_profiler import profile

import jax, scipy

import psutil, os
from pympler.tracker import SummaryTracker

from fastfp.fastfp import Fp_jax
from fastfp.utils import initialize_pta, get_mats

@profile
def FPCalc(psrs, noise, freqs):

    #initialize memory tracking
    #fp_process = psutil.Process(os.getpid())
    #fp_memory_base = fp_process.memory_info().rss

    print('Initialize PTA')
    # initialize the PTA object and precompute a bunch of
    # fixed matrix products that go into the Fp-statistic calculation
    pta = initialize_pta(psrs, noise, inc_cp=True, gwb_comps=30)
    
    #memory tracking
    #fp_memory = fp_process.memory_info().rss
    #print(f'FPCalc Function, PTA generation: {fp_memory - fp_memory_base}')

    print('Precompute matrix wall.')
    t_start = time.perf_counter()
    Nvecs, Ts, sigmainvs = get_mats(pta, noise)
    t_end = time.perf_counter()
    print('Precompute matrix wall time: {0:.4f} s'.format(t_end - t_start))
    
    #memory tracking
    #fp_memory = fp_process.memory_info().rss
    #print(f'FPCalc Function, Matrix Wall: {fp_memory - fp_memory_base}')

    print('initialize Fp-stat class')
    Fp_obj = Fp_jax(psrs, pta)
    
    #memory tracking
    #fp_memory = fp_process.memory_info().rss
    #print(f'FPCalc Function, FP Object: {fp_memory - fp_memory_base}')
    
    # define the range of GW frequencies
    #freqs = np.linspace(2e-9, 1e-7, 200)
    
    # the Fp-statistic calculation is jit-compiled and batched
    # over the array of frequencies, turning it into a single
    # calculation instead of a for loop
    # (for more info, see documentation for jax.vmap)
    t_start = time.perf_counter()
    fn = jax.vmap(Fp_obj.calculate_Fp, in_axes=(0, None, None, None))
    fps = fn(freqs, Nvecs, Ts, sigmainvs)
    t_end = time.perf_counter()
    print('Fp-statistic wall time: {0:.4f} s'.format(t_end - t_start))
    
    #memory tracking
    #fp_memory = fp_process.memory_info().rss
    #print(f'FPCalc Function, FP Statistic: {fp_memory - fp_memory_base}')

    del pta, Nvecs, Ts, sigmainvs, Fp_obj
    
    #memory tracking
    #fp_memory = fp_process.memory_info().rss
    #print(f'FPCalc Function, Attempted Cleanup: {fp_memory - fp_memory_base}')

    return fps
    
@profile
def FAPCalc(N, fp0):
    n = np.arange(0, N)
    return (np.sum(np.exp(n*np.log(fp0)-fp0-np.log(scipy.special.gamma(n+1)))))

@profile
def FAP(primpuls, CW, noise):
    #initialize memory tracking
    #fap_process = psutil.Process(os.getpid())
    #fap_memory_base = fap_process.memory_info().rss

    #generate simulations from primary pulsars
    psrs = GWInjectionSim.generate(primpuls, noise, CW)
    
    #memory tracking
    #fap_memory = fap_process.memory_info().rss
    #print(f'FAP Function, post generation: {fap_memory - fap_memory_base}')

    #calculate fp statistic
    fgw=np.linspace(10**CW[3], 10**CW[3], 1)
    fp = FPCalc(psrs, noise, fgw)
    
    #memory tracking
    #fap_memory = fap_process.memory_info().rss
    #print(f'FAP Function, post FP statistic: {fap_memory - fap_memory_base}')

    #calculate False Alarm Probability
    fap = FAPCalc(len(primpuls), fp)
    
    #memory tracking
    #fap_memory = fap_process.memory_info().rss
    #print(f'FAP Function, post FAP calculation: {fap_memory - fap_memory_base}')

    return psrs, fp, fap

@profile
def SaveResults(CW, ras, dec, fp, fap, pnum, save):
    fgw = (10**(CW[3]+9))
    #filename = f'R{rascension}_D{declination}_h{Decimal(CW[4]):.4}_f{Decimal(fgw):.4}_p{pnum}'
    #with open(save + filename + '.pkl', 'wb') as f:
    #    pickle.dump(psrs, f)

    with open(save + 'FAP_Data.txt', 'a') as g:
        parameters = [ras, dec, CW, fp, fap, pnum]
        g.write(str(parameters)+'/n')
        
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--pnumber', help='How many pulsars to simulate? Will select for longest observation period from data.', type=int, default=14)
    parser.add_argument('-s', '--save', help='Where do you want the data? Defaults to CWD/Data/Sims', default=(str(Path.cwd()) + '/Data/Sims/'))
    parser.add_argument('-d', '--data', help='Where is the data located? Defaults to CWD/Data', default=(str(Path.cwd()) + '/Data/'))
    parser.add_argument('-i', '--inclination', help='Source Inclination in radians.', type=float, default=0)
    parser.add_argument('-mc', '--chirpmass', help='log of Chirp Mass in solar masses', type=float, default=9)
    parser.add_argument('--phase0', help='Initial phase', type=float, default=0)
    parser.add_argument('--psi', help='SOurce Orientation variable', type=float, default=0)
    parser.add_argument('--hmin', help='The lowest h value (log) to start from.', type = float, default=-18)
    parser.add_argument('--hmax', help='The highest h value (log) to start from.', type=float, default=-13)
    parser.add_argument('--rascension', help='Right Ascension of source, in hours.', type=float, default=18)
    parser.add_argument('--declination', help='Declination of source, in degrees.', type=float, default=-15)
    parser.add_argument('--fgw', help='GW Frequency, in Hz.', type=float, default=2e-9)
    parser.add_argument('--threshold', help='The FAP value you are looking to find. Defaults to .05.', type=float, default=.05)
    parser.add_argument('--accuracy', help='How accurate an answer. Defaults to .001 difference from threshold.', type=int, default=.01)

    args = parser.parse_args()

    #General Variables and File Information
    threshold = args.threshold
    accuracy = args.accuracy
    save = args.save
    data = args.data
    pnum = args.pnumber
    with open(data + 'RedNoiseLibrary.json', 'r') as f:
            noise = json.load(f)

    #Continuous Wave Parameters
    cos_i = np.cos(args.inclination)
    cos_theta = np.cos(np.pi * (90-args.declination) / 180)
    Mc = args.chirpmass
    phase0 = args.phase0
    psi = args.psi
    hmin=args.hmin
    hmax=args.hmax
    rascension=args.rascension
    declination=args.declination
    fgwraw = args.fgw
    fglog10 = np.log10(fgwraw)
    fgw=np.linspace(fgwraw, fgwraw, 1)
    phi = rascension/24*360

    #Populate list of pulsars
    primpuls = DailyAvgPop(data)

    """
    #DEBUG: Track Memory usage
    tracker = SummaryTracker()
    tracker.print_diff()
    process = psutil.Process(os.getpid())
    memory_base = process.memory_info().rss
    """

    #Check hmin and hmax before entering the loop.
    CW = [cos_i, cos_theta, Mc, fglog10, hmin, phase0, phi, psi]
    _, fpmin, fapmin = FAP(primpuls, CW, noise)
    SaveResults(CW, rascension, declination, fpmin, fapmin, pnum, save)
    print(f'fpmin is type {type(fpmin)} and {fpmin}')
    print(f'fapmin is type {type(fpmin)} and {fapmin}')
    #print(f'psrs is {type(psrs)}')

    """
    #DEBUG:Memory tracking
    memory = process.memory_info().rss
    print(f'Using {(memory - memory_base)*1e-9}GB RAM after hmin test.')
    tracker.print_diff()
    #SaveResults(psrs, CW, rascension, declination, fpmin, fapmin, pnum, save + 'Test/')
    """

    if (fapmin - threshold) < accuracy and fapmin > threshold:
        print('saving at hmin')
        SaveResults(CW, rascension, declination, fpmax, fapmax, pnum, save)

    
    else:
        print('hmin was outside of bounds, checking hmax')
        CW[4] = hmax
        _, fpmax, fapmax = FAP(primpuls, CW, noise)
        SaveResults(CW, rascension, declination, fpmax, fapmax, pnum, save)
        print(fpmax)
        print(fapmax)

        """
        #DEBUG: Memory tracking
        memory = process.memory_info().rss
        print(f'Using {(memory - memory_base)*1e-9}GB RAM after hmax test.')
        tracker.print_diff()
        #SaveResults(psrs, CW, rascension, declination, fpmax, fapmax, pnum, save + 'Test/')
        """

        if (threshold - fapmax) < accuracy and fapmax < threshold:
            SaveResults(CW, rascension, declination, fpmin, fapmin, pnum, save)
        
        elif fapmax < threshold and fapmin > threshold:
            print('hmin and hmax range is ok')
            count = 0
            hmid = (hmin+hmax)/2
            CW[4] = hmid
            _, fpmid, fapmid = FAP(primpuls, CW, noise)
            SaveResults(CW, rascension, declination, fpmid, fapmid, pnum, save)
            print(fpmid)
            print(fapmid)
            
            while ((fapmid - threshold) > accuracy) or ((threshold - fapmid) > accuracy):               
                count += 1
                hmid = (hmin+hmax)/2
                CW[4] = hmid
                _, fpmid, fapmid = FAP(primpuls, CW, noise)
                SaveResults(CW, rascension, declination, fpmid, fapmid, pnum, save)
                print(f'Count: {count}')
                print(fpmid)
                print(fapmid)

                """
                #DEBUG: Memory Tracking
                memory = process.memory_info().rss
                print(f'Bisecting count: {count} is using {(memory - memory_base)*1e-9}GB RAM after simulation')
                tracker.print_diff()
                """
    
                if (fapmid - threshold) > accuracy:
                    hmin = hmid
                elif (threshold - fapmid) > accuracy:
                    hmax = hmid
                
            print(f'saving results at h: {Decimal(hmid):.4}')
            SaveResults(CW, rascension, declination, fpmid, fapmid, pnum, save)

        else:
            print('There was an error.')
