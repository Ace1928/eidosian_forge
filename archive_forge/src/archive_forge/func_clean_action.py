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
def clean_action():
    """Remove every profile file in current directory."""
    parser = ArgumentParser(usage='mprof clean\nThis command takes no argument.')
    parser.add_argument('--version', action='version', version=mp.__version__)
    parser.add_argument('--dry-run', dest='dry_run', default=False, action='store_true', help='Show what will be done, without actually doing it.')
    args = parser.parse_args()
    filenames = get_profile_filenames('all')
    if args.dry_run:
        print('Files to be removed: ')
        for filename in filenames:
            print(filename)
    else:
        for filename in filenames:
            os.remove(filename)