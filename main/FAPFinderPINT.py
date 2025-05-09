#!/usr/bin/env python

import GWInjectionSim

import numpy as np
from pathlib import Path
import time, json, pickle, utility
from decimal import Decimal

import datetime

import jax, scipy
import random

import enterprise

from fastfp.fastfp import FastFp
from fastfp.utils import initialize_pta, get_mats_fp


def FPCalc(psrs, noise, freqs):

    '''
    param: psrs: list of pulsars
    param: noise: dictionary of noise parameters
    param: freqs: array of frequencies (size 1) in decimal form

    return: fps: array of Fp statistics

    Uses Gabe Freedman's FastFp package to calculate the Fp statistic
    '''

    print("Initialize PTA")
    # initialize the PTA object and precompute a bunch of
    # fixed matrix products that go into the Fp-statistic calculation
    pta = initialize_pta(psrs, noise, inc_cp=True, gwb_comps=30)

    print("Precompute matrix wall.")
    t_start = time.perf_counter()
    Nvecs, Ts, sigmainvs = get_mats_fp(pta, noise)
    t_end = time.perf_counter()
    print("Precompute matrix wall time: {0:.4f} s".format(t_end - t_start))

    print("initialize Fp-stat class")
    Fp_obj = FastFp(psrs, pta)

    # the Fp-statistic calculation is jit-compiled and batched
    # over the array of frequencies, turning it into a single
    # calculation instead of a for loop
    # (for more info, see documentation for jax.vmap)
    t_start = time.perf_counter()
    fn = jax.vmap(Fp_obj.calculate_Fp, in_axes=(0, None, None, None))
    fps = fn(freqs, Nvecs, Ts, sigmainvs)
    t_end = time.perf_counter()
    print("Fp-statistic wall time: {0:.4f} s".format(t_end - t_start))

    del pta, Nvecs, Ts, sigmainvs, Fp_obj

    return fps


def FAPCalc(N, fp0):
    '''
    param: N: number of pulsars
    param: fp0: Fp statistic

    return: fap: False Alarm Probability
    
    Uses the Fp statistic to calculate the False Alarm Probability'''

    n = np.arange(0, N)
    return np.sum(np.exp(n * np.log(fp0) - fp0 - np.log(scipy.special.gamma(n + 1))))


def FAP(primpuls, CW, noise):

    '''
    param: primpuls: list of pulsars
    param: CW: list of continuous wave parameters
    param: noise: dictionary of noise parameters

    return: fp: Fp statistic (array of size 1)
    return: fap: False Alarm Probability

    Injects the received CW into the primary pulsar list and creates a secondary pulsar list.
    Calculates the Fp statistic and the False Alarm Probability.
    '''

    # Inject CW simulations into primary pulsars
    # Creates a secondary pulsar list
    psrs = GWInjectionSim.generate(primpuls, noise, CW)

    # calculate fp statistic
    # fgw is an array of size 1 because FastFP expects an array
    fgw = np.linspace(10 ** CW[3], 10 ** CW[3], 1)
    fp = FPCalc(psrs, noise, fgw)

    # calculate False Alarm Probability
    fap = FAPCalc(len(primpuls), fp)

    return fp, fap


def SaveResults(CW, ras, dec, fp, fap, pnum, save, searchtype):
    '''
    param: CW: list of continuous wave parameters
    param: ras: right ascension of source
    param: dec: declination of source
    param: fp: Fp statistic
    param: fap: False Alarm Probability
    param: pnum: number of pulsars
    param: save: file path of where to save the data
    param: searchtype: Either None (Grid) or the threshold value (Bisection)

    Adds a line to the defined file with all the passed parameters.
    Will not overwrite, to allow for multiple runs,
    but will create a file if it doesn't exist.
    '''
    # Convert to decimal for human readability
    fgw = 10 ** (CW[3] + 9)

    # filename can be used to auto sort files by parameters if desired
    filename = f"R{ras}_D{dec}_f{Decimal(fgw):.4}_p{pnum}"

    # Grid Search sends None for searchtype
    # each data entry will be labeled 'Grid' and include N/A for threshold
    if searchtype is None:
        with open(save + filename + "FAP_Data.txt", "a") as g:
            parameters = ['Grid', 'N/A', ras, dec, CW, fp[0], fap, pnum]
            g.write(str(parameters) + "\n")

    # Bisection search sends the threshold value for searchtype
    # Each data entry will be labeled 'Bisect' and include the threshold value       
    else:
        with open(save + filename + "FAP_Data.txt", "a") as g:
            parameters = ['Bisect', searchtype, ras, dec, CW, fp[0], fap, pnum]
            g.write(str(parameters) + "\n")

    return None


def Bisection(primpuls, CW, hmin, hmax, threshold, noise):

    '''
    param: primpuls: primary list of pulsars
    param: CW: list of continuous wave parameters
    param: hmin: minimum h value (log)
    param: hmax: maximum h value (log)
    param: threshold: FAP value search parameter
    param: noise: dictionary of noise parameters

    Uses the Bisection method to find the h value that gives the desired FAP value.
    Due to statistical variance, It can wobble around the target value.
    For stability, the code will loop a maximum of 10 times.
    '''

    # Check hmin before entering the loop.
    accuracy = threshold / 10

    # Randomize phase0 if not defined
    if CW[5] is None:
        CW[5] = 2 * np.pi * random.random()

    # redefine appropriate CW parameters
    CW[4] = hmin

    # Calculate the Fp statistic and FAP
    fpmin, fapmin = FAP(primpuls, CW, noise)
    print(f"fpmin is {fpmin}")
    print(f"fapmin is {fapmin}")

    # Save the results
    SaveResults(
        CW, rascension, declination, fpmin, fapmin, len(primpuls), save, threshold
    )

    # Check if fapmin is within the desired range
    if abs(fapmin - threshold) < accuracy and fapmin > threshold - accuracy:
        print("hmin fell within tolerance.")

    # Check hmax before entering the loop.
    else:
        print("hmin was outside of bounds, checking hmax")

        # Randomize phase0 if not defined
        if CW[5] is None:
            CW[5] = 2 * np.pi * random.random()

        # redefine appropriate CW parameters
        CW[4] = hmax

        # Calculate the Fp statistic and FAP
        fpmax, fapmax = FAP(primpuls, CW, noise)
        print(f"fpmax is {fpmax}")
        print(f"fapmax is {fapmax}")

        # Save the results
        SaveResults(
            CW, rascension, declination, fpmax, fapmax, len(primpuls), save, threshold
        )

        # Check if fapmax is within the desired range
        if abs(threshold - fapmax) < accuracy and fapmax < threshold + accuracy:
            print("Stopping at fapmax")

        # hmin should have a high fap, and hmax should have a low fap
        # This skips the loop if they are on the wrong side of the threshold
        elif fapmax < threshold and fapmin > threshold:

            # Perform one bisection to define hmid before looping
            # This was a bit awkward, might benefit from some rearrangement
            print("hmin and hmax range is ok")
            count = 0
            hmid = (hmin + hmax) / 2

            # Randomize phase0 if not defined
            if CW[5] is None:
                CW[5] = 2 * np.pi * random.random()

            # redefine appropriate CW parameters
            CW[4] = hmid

            # Calculate the Fp statistic and FAP
            fpmid, fapmid = FAP(primpuls, CW, noise)
            print(f"fpmid is {fpmid}")
            print(f"fapmid is {fapmid}")

            # Save the results
            SaveResults(
                CW, rascension, declination, fpmid, fapmid, len(primpuls), save, threshold
            )

            # redefine hmin or hmax based on the fapmid value
            if (fapmid - threshold) > accuracy:
                hmin = hmid
            elif (threshold - fapmid) > accuracy:
                hmax = hmid

            # Loop until the fapmid value is within the desired range
            # or the count reaches 10
            while (
                ((fapmid - threshold) > accuracy) or ((threshold - fapmid) > accuracy)
            ) and count < 10:
                count += 1
                hmid = (hmin + hmax) / 2

                # Randomize phase0 if not defined
                if CW[5] is None:
                    CW[5] = 2 * np.pi * random.random()

                # redefine appropriate CW parameters
                CW[4] = hmid

                # Calculate the Fp statistic and FAP
                fpmid, fapmid = FAP(primpuls, CW, noise)
                print(f"fpmid is {fpmid}")
                print(f"fapmid is {fapmid}")

                # Save the results
                SaveResults(
                    CW, rascension, declination, fpmid, fapmid, len(primpuls), save, threshold
                )
                print(f"Count: {count}")

                # redefine hmin or hmax based on the fapmid value
                if (fapmid - threshold) > accuracy:
                    hmin = hmid
                elif (threshold - fapmid) > accuracy:
                    hmax = hmid

            print("Task Complete")
            # SaveResults(CW, rascension, declination, fpmid, fapmid, len(primpuls), save)

        # If the hmin and hmax values are on the wrong side of the threshold
        # this will print an error message
        else:
            print("There was an error.")

    return None


def Grid(primpuls, CW, hmin, hmax, noise):

    '''
    param: primpuls: primary list of pulsars
    param: CW: list of continuous wave parameters
    param: hmin: minimum h value (log)
    param: hmax: maximum h value (log)
    param: noise: dictionary of noise parameters

    Divides the range of h values into 10 equal parts
    and calculates the Fp statistic and FAP for each value.
    '''
    range = np.linspace(hmin, hmax, 10)
    for i in range:

        # Randomize phase0 if not defined
        if CW[5] is None:
            phase0 = 2 * np.pi * random.random()

        # redefine appropriate CW parameters
        CW[4] = i
        CW[5] = phase0

        # Calculate the Fp statistic and FAP
        fp, fap = FAP(primpuls, CW, noise)
        print(f"fp is {fp}")
        print(f"fap is {fap}")
        print(i)
        SaveResults(CW, rascension, declination, fp, fap, len(primpuls), save, None)

    return None


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save",
        help="Where do you want the results? Defaults to CWD/Data/Sims",
        default=(str(Path.cwd()) + "/Data/Sims/"),
    )
    parser.add_argument(
        "-d",
        "--data",
        help="Where is the data located? Defaults to CWD/Data",
        default=(str(Path.cwd()) + "/Data/"),
    )
    parser.add_argument(
        "-i",
        "--inclination",
        help="Source Inclination in radians.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-mc",
        "--chirpmass",
        help="log of Chirp Mass in solar masses",
        type=float,
        default=9,
    )
    parser.add_argument(
        '--phase0', 
        help='Initial phase. If undefined, will be randomized with each simulation',
        type=float,
        default=None
    )
    parser.add_argument(
        "--psi",
        help="Source Orientation variable",
        type=float,
        default=0
    )
    parser.add_argument(
        "--hmin",
        help="The lowest h value (log) to start from.",
        type=float,
        default=-18,
    )
    parser.add_argument(
        "--hmax",
        help="The highest h value (log) to start from.",
        type=float,
        default=-13,
    )
    parser.add_argument(
        "--rascension",
        help="Right Ascension of source, in hours.",
        type=float,
        default=18,
    )
    parser.add_argument(
        "--declination",
        help="Declination of source, in degrees.",
        type=float,
        default=-15,
    )
    parser.add_argument(
        "--fgw",
        help="GW Frequency, in Hz.",
        type=float,
        default=2e-9
    )
    parser.add_argument(
        "--threshold",
        help="The FAP value you are looking to find. Defaults to .001.",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "-rpt",
        "--repeat",
        help="How many repetitions to run. CURRENTLY UNSTABLE",
        type=int,
        default=1
    )
    parser.add_argument(
        "--ptafile",
        help="If undefined, will create a pta. Can be defined to use an existing pickle file.",
        default=None,
    )
    parser.add_argument(
        "--timeline",
        help="Define the end date for generated observations.",
        type=float,
        default=60000,
    )
    parser.add_argument(
        "--searchtype", help="Bisection or Grid search", default="Bisection"
    )

    args = parser.parse_args()

    # General Variables and File Information
    threshold = args.threshold
    save = args.save
    data = args.data
    with open(data + "RedNoiseLibrary.json", "r") as f:
        noise = json.load(f)
    repeat = args.repeat
    ptafile = args.ptafile
    timestamp = str(datetime.datetime.now().timestamp()).split(".")[0]
    timeline = args.timeline
    searchtype = args.searchtype

    # Continuous Wave Parameters
    cos_i = np.cos(args.inclination)
    cos_theta = np.cos(np.pi * (90 - args.declination) / 180)
    Mc = args.chirpmass
    psi = args.psi
    hmin = args.hmin
    hmax = args.hmax
    rascension = args.rascension
    declination = args.declination
    fgwraw = args.fgw
    fglog10 = np.log10(fgwraw)
    fgw = np.linspace(fgwraw, fgwraw, 1)
    phi = rascension / 24 * 360
    phase0 = args.phase0

    # Continuous Wave Parameter List
    CW = [cos_i, cos_theta, Mc, fglog10, hmin, phase0, phi, psi]

    # Populate list of pulsars if no pta file is defined
    if ptafile is None:
        temp = utility.DailyAvgPop(data, timeline)

        primpuls = []
        for psr in temp:
            primpuls.append(enterprise.pulsar.Pulsar(psr.toas, psr.model))

        # Pickle the PTA object for later use
        with open(save + timestamp + "Pulsars.pkl", mode="xb") as pkl:
            pickle.dump(primpuls, pkl)

    # If a pta file is defined, load the PTA object from the file
    else:
        with open(ptafile, mode="rb") as pkl:
            primpuls = pickle.load(pkl)

    """
    Repeat crashes the code if you repeat more than 10 or 15 times
    """

    if searchtype == "Grid":
        for i in range(repeat):
            Grid(primpuls, CW, hmin, hmax, noise)

    elif searchtype == "Bisection":
        for i in range(repeat):
            Bisection(primpuls, CW, hmin, hmax, threshold, noise)

    else:
        print("searchtype must be either Grid or Bisection")
