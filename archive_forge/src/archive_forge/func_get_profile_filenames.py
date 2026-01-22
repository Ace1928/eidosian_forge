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
def get_profile_filenames(args):
    """Return list of profile filenames.

    Parameters
    ==========
    args (list)
        list of filename or integer. An integer is the index of the
        profile in the list of existing profiles. 0 is the oldest,
        -1 in the more recent.
        Non-existing files cause a ValueError exception to be thrown.

    Returns
    =======
    filenames (list)
        list of existing memory profile filenames. It is guaranteed
        that an given file name will not appear twice in this list.
    """
    profiles = glob.glob('mprofile_??????????????.dat')
    profiles.sort()
    if args == 'all':
        filenames = copy.copy(profiles)
    else:
        filenames = []
        for arg in args:
            if arg == '--':
                continue
            try:
                index = int(arg)
            except ValueError:
                index = None
            if index is not None:
                try:
                    filename = profiles[index]
                except IndexError:
                    raise ValueError('Invalid index (non-existing file): %s' % arg)
                if filename not in filenames:
                    filenames.append(filename)
            elif osp.isfile(arg):
                if arg not in filenames:
                    filenames.append(arg)
            elif osp.isdir(arg):
                raise ValueError('Path %s is a directory' % arg)
            else:
                raise ValueError('File %s not found' % arg)
    for filename in reversed(filenames):
        parts = osp.splitext(filename)
        timestamp_file = parts[0] + '_ts' + parts[1]
        if osp.isfile(timestamp_file) and timestamp_file not in filenames:
            filenames.append(timestamp_file)
    return filenames