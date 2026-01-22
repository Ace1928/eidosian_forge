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
def plot_action():

    def xlim_type(value):
        try:
            newvalue = [float(x) for x in value.split(',')]
        except:
            raise ArgumentError("'%s' option must contain two numbers separated with a comma" % value)
        if len(newvalue) != 2:
            raise ArgumentError("'%s' option must contain two numbers separated with a comma" % value)
        return newvalue
    desc = 'Plots using matplotlib the data file `file.dat` generated\nusing `mprof run`. If no .dat file is given, it will take the most recent\nsuch file in the current directory.'
    parser = ArgumentParser(usage='mprof plot [options] [file.dat]', description=desc)
    parser.add_argument('--version', action='version', version=mp.__version__)
    parser.add_argument('--title', '-t', dest='title', default=None, type=str, action='store', help='String shown as plot title')
    parser.add_argument('--no-function-ts', '-n', dest='no_timestamps', action='store_true', help='Do not display function timestamps on plot.')
    parser.add_argument('--output', '-o', help='Save plot to file instead of displaying it.')
    parser.add_argument('--window', '-w', dest='xlim', type=xlim_type, help='Plot a time-subset of the data. E.g. to plot between 0 and 20.5 seconds: --window 0,20.5')
    parser.add_argument('--flame', '-f', dest='flame_mode', action='store_true', help='Plot the timestamps as a flame-graph instead of the default brackets')
    parser.add_argument('--slope', '-s', dest='slope', action='store_true', help='Plot a trend line and its numerical slope')
    parser.add_argument('--backend', help='Specify the Matplotlib backend to use')
    parser.add_argument('profiles', nargs='*', help='profiles made by mprof run')
    args = parser.parse_args()
    try:
        if args.backend is not None:
            import matplotlib
            matplotlib.use(args.backend)
        import pylab as pl
    except ImportError as e:
        print('matplotlib is needed for plotting.')
        print(e)
        sys.exit(1)
    pl.ioff()
    filenames = get_profiles(args)
    fig = pl.figure(figsize=(14, 6), dpi=90)
    if not args.flame_mode:
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    else:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    if args.xlim is not None:
        pl.xlim(args.xlim[0], args.xlim[1])
    if len(filenames) > 1 or args.no_timestamps:
        timestamps = False
    else:
        timestamps = True
    plotter = plot_file
    if args.flame_mode:
        plotter = flame_plotter
    for n, filename in enumerate(filenames):
        mprofile = plotter(filename, index=n, timestamps=timestamps, options=args)
    pl.xlabel('time (in seconds)')
    pl.ylabel('memory used (in MiB)')
    if args.title is None and len(filenames) == 1:
        pl.title(mprofile['cmd_line'])
    elif args.title is not None:
        pl.title(args.title)
    if not args.flame_mode:
        leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        leg.get_frame().set_alpha(0.5)
        pl.grid()
    if args.output:
        pl.savefig(args.output)
    else:
        pl.show()