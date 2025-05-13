FAPFinder simulates pulsars using raw pulsar data and calculates the FP and FAP values.

FAPFinderPINT.py is the primary run file.

Dependancies:
numpy
pathlib
time
json
pickle
decimal
datetime
jax
scipy
random
enterprise
fastfp

Command Line Arguments:

    "-s",
    "--save",
    String, Default: [CWD] + "/Data/Sims/"
    Location for saving results. If a file exists, it will append to the file.

    "-d",
    "--data",
    String, Default: [CWD] + "/Data/"
    The location of your data files. If generating a new PTA, it will look for a "par" and "tim" folder in the data directory. It also expects a "RedNoiseLibrary.json" to be in the data directory.

    "-i",
    "--inclination",
    Float, Default: 0
    Source Inclination in radians.

    "-mc",
    "--chirpmass",
    Float, Default: 9
    Log of Chirp Mass in solar masses.

    "--phase0",
    Float, Default: None
    Initial phase. If undefined, will be randomized with each simulation.

    "--psi",
    Float, Default: 0
    Source Orientation variable

    "--hmin",
    Float, Default: -18
    The log of the lower bound on strain h.

    "--hmax",
    Float, Default: -13
    The log of the upper bound on strain h.

    "--rascension",
    Float, Default: 18
    Right Ascension of GW source, in hours.

    "--declination",
    Float, Default: -15
    Declination of GW source, in degrees.

    "--fgw",
    Float, Default: 2e-9
    GW Frequency, in Hz.

    "--threshold",
    Float, Default: .001
    Used as a guide in a bisection search, threshold is the FAP value the search will try to narrow in on
    while varying the strain h.

    "-rpt",
    "--repeat",
    Integer, Default: 1
    This argument exists to allow the program to loop a simulation multiple times.
    Currently the code tends to memory fault if running more than 15ish simulations in a row.
    Recommended not to use repeat unless a fix is found.

    "--ptafile",
    String, Default: None
    If undefined, the program will search the data directory for par and tim directories to create a new pta.
    New PTAs will be saved as a pickle file.
    --ptafile can be used to define a path to an existing pickle file, which will be loaded as a PTA instead of creating one.

    "--timeline",
    Float, Default: 60000
    Define a time, in MJD, to extend observations out to. The program will use each existing raw pulsar file to statistically generate a similar pulsar, and extend the observation time out to this date.

    "--searchtype", 
    String, Default: "Bisection"
    Defines whether to perform a bisection or a grid search. Bisection will try to narrow in on a strain h that meets the --threshold. Grid search will run 10 evenly spaced simulations across the range of --hmin and --hmax


Saved File Format:

The program appends a string to the end of the given save file of the following format:

[Search Type, Threshold, Right Ascension, Declination, CW, FP Statistic, False Alarm Probability, # of Pulsars]\n

Where CW is a list of the Continuous Wave parameters:

[cosine of Inclination, cos of Theta (source location), Chirp Mass, Log of wave frequency, strain h, Initial Phase, phi (source location), psi]

The \n is for human readability and for separating the entries in code.

Credit:

This code was organized and assembled by David Marsella. The time averaged and statistical generation of pulsars was written by Sarah J. Vigeland. FastFP is made by Gabe Freedman.



