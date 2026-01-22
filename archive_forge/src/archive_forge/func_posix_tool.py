from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
def posix_tool():
    if include_children:
        raise NotImplementedError('The psutil module is required to monitor the memory usage of child processes.')
    warnings.warn('psutil module not found. memory_profiler will be slow')
    out = subprocess.Popen(['ps', 'v', '-p', str(pid)], stdout=subprocess.PIPE).communicate()[0].split(b'\n')
    try:
        vsz_index = out[0].split().index(b'RSS')
        mem = float(out[1].split()[vsz_index]) / 1024
        if timestamps:
            return (mem, time.time())
        else:
            return mem
    except:
        if timestamps:
            return (-1, time.time())
        else:
            return -1