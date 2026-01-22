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
def simple_cythonize(src, destdir=None, cwd=None, **cy_kwargs):
    """ Generates a C file from a Cython source file.

    Parameters
    ==========

    src: str
        Path to Cython source.
    destdir: str (optional)
        Path to output directory (default: '.').
    cwd: path string (optional)
        Root of relative paths (default: '.').
    **cy_kwargs:
        Second argument passed to cy_compile. Generates a .cpp file if ``cplus=True`` in ``cy_kwargs``,
        else a .c file.
    """
    from Cython.Compiler.Main import default_options, CompilationOptions
    from Cython.Compiler.Main import compile as cy_compile
    assert src.lower().endswith('.pyx') or src.lower().endswith('.py')
    cwd = cwd or '.'
    destdir = destdir or '.'
    ext = '.cpp' if cy_kwargs.get('cplus', False) else '.c'
    c_name = os.path.splitext(os.path.basename(src))[0] + ext
    dstfile = os.path.join(destdir, c_name)
    if cwd:
        ori_dir = os.getcwd()
    else:
        ori_dir = '.'
    os.chdir(cwd)
    try:
        cy_options = CompilationOptions(default_options)
        cy_options.__dict__.update(cy_kwargs)
        if 'language_level' not in cy_kwargs:
            cy_options.__dict__['language_level'] = 3
        cy_result = cy_compile([src], cy_options)
        if cy_result.num_errors > 0:
            raise ValueError('Cython compilation failed.')
        if os.path.realpath(os.path.dirname(src)) != os.path.realpath(destdir):
            if os.path.exists(dstfile):
                os.unlink(dstfile)
            shutil.move(os.path.join(os.path.dirname(src), c_name), destdir)
    finally:
        os.chdir(ori_dir)
    return dstfile