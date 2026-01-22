import os
import sys
import importlib.util
import py_compile
import struct
import filecmp
from functools import partial
from pathlib import Path
def compile_dir(dir, maxlevels=None, ddir=None, force=False, rx=None, quiet=0, legacy=False, optimize=-1, workers=1, invalidation_mode=None, *, stripdir=None, prependdir=None, limit_sl_dest=None, hardlink_dupes=False):
    """Byte-compile all modules in the given directory tree.

    Arguments (only dir is required):

    dir:       the directory to byte-compile
    maxlevels: maximum recursion level (default `sys.getrecursionlimit()`)
    ddir:      the directory that will be prepended to the path to the
               file as it is compiled into each byte-code file.
    force:     if True, force compilation, even if timestamps are up-to-date
    quiet:     full output with False or 0, errors only with 1,
               no output with 2
    legacy:    if True, produce legacy pyc paths instead of PEP 3147 paths
    optimize:  int or list of optimization levels or -1 for level of
               the interpreter. Multiple levels leads to multiple compiled
               files each with one optimization level.
    workers:   maximum number of parallel workers
    invalidation_mode: how the up-to-dateness of the pyc will be checked
    stripdir:  part of path to left-strip from source file path
    prependdir: path to prepend to beginning of original file path, applied
               after stripdir
    limit_sl_dest: ignore symlinks if they are pointing outside of
                   the defined path
    hardlink_dupes: hardlink duplicated pyc files
    """
    ProcessPoolExecutor = None
    if ddir is not None and (stripdir is not None or prependdir is not None):
        raise ValueError('Destination dir (ddir) cannot be used in combination with stripdir or prependdir')
    if ddir is not None:
        stripdir = dir
        prependdir = ddir
        ddir = None
    if workers < 0:
        raise ValueError('workers must be greater or equal to 0')
    if workers != 1:
        from concurrent.futures.process import _check_system_limits
        try:
            _check_system_limits()
        except NotImplementedError:
            workers = 1
        else:
            from concurrent.futures import ProcessPoolExecutor
    if maxlevels is None:
        maxlevels = sys.getrecursionlimit()
    files = _walk_dir(dir, quiet=quiet, maxlevels=maxlevels)
    success = True
    if workers != 1 and ProcessPoolExecutor is not None:
        workers = workers or None
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(partial(compile_file, ddir=ddir, force=force, rx=rx, quiet=quiet, legacy=legacy, optimize=optimize, invalidation_mode=invalidation_mode, stripdir=stripdir, prependdir=prependdir, limit_sl_dest=limit_sl_dest, hardlink_dupes=hardlink_dupes), files)
            success = min(results, default=True)
    else:
        for file in files:
            if not compile_file(file, ddir, force, rx, quiet, legacy, optimize, invalidation_mode, stripdir=stripdir, prependdir=prependdir, limit_sl_dest=limit_sl_dest, hardlink_dupes=hardlink_dupes):
                success = False
    return success