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
def exec_with_profiler(filename, profiler, backend, passed_args=[]):
    from runpy import run_module
    builtins.__dict__['profile'] = profiler
    ns = dict(_CLEAN_GLOBALS, profile=profiler, __file__=filename)
    sys.path.insert(0, os.path.dirname(script_filename))
    _backend = choose_backend(backend)
    sys.argv = [filename] + passed_args
    try:
        if _backend == 'tracemalloc' and has_tracemalloc:
            tracemalloc.start()
        with io.open(filename, encoding='utf-8') as f:
            exec(compile(f.read(), filename, 'exec'), ns, ns)
    finally:
        if has_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()