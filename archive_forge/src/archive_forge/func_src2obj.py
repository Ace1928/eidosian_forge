import glob
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from sysconfig import get_config_var, get_config_vars, get_path
from .runners import (
from .util import (
def src2obj(srcpath, Runner=None, objpath=None, cwd=None, inc_py=False, **kwargs):
    """ Compiles a source code file to an object file.

    Files ending with '.pyx' assumed to be cython files and
    are dispatched to pyx2obj.

    Parameters
    ==========

    srcpath: str
        Path to source file.
    Runner: CompilerRunner subclass (optional)
        If ``None``: deduced from extension of srcpath.
    objpath : str (optional)
        Path to generated object. If ``None``: deduced from ``srcpath``.
    cwd: str (optional)
        Working directory and root of relative paths. If ``None``: current dir.
    inc_py: bool
        Add Python include path to kwarg "include_dirs". Default: False
    \\*\\*kwargs: dict
        keyword arguments passed to Runner or pyx2obj

    """
    name, ext = os.path.splitext(os.path.basename(srcpath))
    if objpath is None:
        if os.path.isabs(srcpath):
            objpath = '.'
        else:
            objpath = os.path.dirname(srcpath)
            objpath = objpath or '.'
    if os.path.isdir(objpath):
        objpath = os.path.join(objpath, name + objext)
    include_dirs = kwargs.pop('include_dirs', [])
    if inc_py:
        py_inc_dir = get_path('include')
        if py_inc_dir not in include_dirs:
            include_dirs.append(py_inc_dir)
    if ext.lower() == '.pyx':
        return pyx2obj(srcpath, objpath=objpath, include_dirs=include_dirs, cwd=cwd, **kwargs)
    if Runner is None:
        Runner, std = extension_mapping[ext.lower()]
        if 'std' not in kwargs:
            kwargs['std'] = std
    flags = kwargs.pop('flags', [])
    needed_flags = ('-fPIC',)
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)
    run_linker = kwargs.pop('run_linker', False)
    if run_linker:
        raise CompileError('src2obj called with run_linker=True')
    runner = Runner([srcpath], objpath, include_dirs=include_dirs, run_linker=run_linker, cwd=cwd, flags=flags, **kwargs)
    runner.run()
    return objpath