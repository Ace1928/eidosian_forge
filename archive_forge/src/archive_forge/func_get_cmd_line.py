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
def get_cmd_line(args):
    """Given a set or arguments, compute command-line."""
    blanks = set(' \t')
    args = [s if blanks.isdisjoint(s) else "'" + s + "'" for s in args]
    return ' '.join(args)