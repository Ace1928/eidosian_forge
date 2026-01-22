import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def print_cycles(objects, outstream=sys.stdout, show_progress=False):
    """
    Print loops of cyclic references in the given *objects*.

    It is often useful to pass in ``gc.garbage`` to find the cycles that are
    preventing some objects from being garbage collected.

    Parameters
    ----------
    objects
        A list of objects to find cycles in.
    outstream
        The stream for output.
    show_progress : bool
        If True, print the number of objects reached as they are found.
    """
    import gc

    def print_path(path):
        for i, step in enumerate(path):
            next = path[(i + 1) % len(path)]
            outstream.write('   %s -- ' % type(step))
            if isinstance(step, dict):
                for key, val in step.items():
                    if val is next:
                        outstream.write(f'[{key!r}]')
                        break
                    if key is next:
                        outstream.write(f'[key] = {val!r}')
                        break
            elif isinstance(step, list):
                outstream.write('[%d]' % step.index(next))
            elif isinstance(step, tuple):
                outstream.write('( tuple )')
            else:
                outstream.write(repr(step))
            outstream.write(' ->\n')
        outstream.write('\n')

    def recurse(obj, start, all, current_path):
        if show_progress:
            outstream.write('%d\r' % len(all))
        all[id(obj)] = None
        referents = gc.get_referents(obj)
        for referent in referents:
            if referent is start:
                print_path(current_path)
            elif referent is objects or isinstance(referent, types.FrameType):
                continue
            elif id(referent) not in all:
                recurse(referent, start, all, current_path + [obj])
    for obj in objects:
        outstream.write(f'Examining: {obj!r}\n')
        recurse(obj, obj, {}, [])