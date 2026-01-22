import glob
import os
import os.path as osp
import sys
import re
import copy
import time
import math
import logging
import itertools
from ast import literal_eval
from collections import defaultdict
from argparse import ArgumentParser, ArgumentError, REMAINDER, RawTextHelpFormatter
import importlib
import memory_profiler as mp
def peak_action():
    desc = 'Prints the peak memory used in data file `file.dat` generated\nusing `mprof run`. If no .dat file is given, it will take the most recent\nsuch file in the current directory.'
    parser = ArgumentParser(usage='mprof peak [options] [file.dat]', description=desc)
    parser.add_argument('profiles', nargs='*', help='profiles made by mprof run')
    parser.add_argument('--func', dest='func', default=None, help='Show the peak for this function. Does not support child processes.')
    args = parser.parse_args()
    filenames = get_profiles(args)
    for filename in filenames:
        prof = read_mprofile_file(filename)
        try:
            mem_usage = filter_mprofile_mem_usage_by_function(prof, args.func)
        except ValueError:
            print('{}\tNaN MiB'.format(prof['filename']))
            continue
        print('{}\t{:.3f} MiB'.format(prof['filename'], max(mem_usage)))
        for child, values in prof['children'].items():
            child_peak = max([mem_ts[0] for mem_ts in values])
            print('  Child {}\t\t\t{:.3f} MiB'.format(child, child_peak))