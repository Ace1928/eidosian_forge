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
def read_mprofile_file(filename):
    """Read an mprofile file and return its content.

    Returns
    =======
    content: dict
        Keys:

        - "mem_usage": (list) memory usage values, in MiB
        - "timestamp": (list) time instant for each memory usage value, in
            second
        - "func_timestamp": (dict) for each function, timestamps and memory
            usage upon entering and exiting.
        - 'cmd_line': (str) command-line ran for this profile.
    """
    func_ts = {}
    mem_usage = []
    timestamp = []
    children = defaultdict(list)
    cmd_line = None
    f = open(filename, 'r')
    for l in f:
        if l == '\n':
            raise ValueError('Sampling time was too short')
        field, value = l.split(' ', 1)
        if field == 'MEM':
            values = value.split(' ')
            mem_usage.append(float(values[0]))
            timestamp.append(float(values[1]))
        elif field == 'FUNC':
            values = value.split(' ')
            f_name, mem_start, start, mem_end, end = values[:5]
            ts = func_ts.get(f_name, [])
            to_append = [float(start), float(end), float(mem_start), float(mem_end)]
            if len(values) >= 6:
                stack_level = values[5]
                to_append.append(int(stack_level))
            ts.append(to_append)
            func_ts[f_name] = ts
        elif field == 'CHLD':
            values = value.split(' ')
            chldnum = values[0]
            children[chldnum].append((float(values[1]), float(values[2])))
        elif field == 'CMDLINE':
            cmd_line = value
        else:
            pass
    f.close()
    return {'mem_usage': mem_usage, 'timestamp': timestamp, 'func_timestamp': func_ts, 'filename': filename, 'cmd_line': cmd_line, 'children': children}