import os
import re
import shutil
import sys
import tempfile
import zipfile
from glob import glob
from os.path import abspath
from os.path import join as pjoin
from subprocess import PIPE, Popen
import os
import sys
import {mod_name}
def run_mod_cmd(mod_name, pkg_path, cmd, script_dir=None, print_location=True):
    """Run command in own process in anonymous path

    Parameters
    ----------
    mod_name : str
        Name of module to import - e.g. 'nibabel'
    pkg_path : str
        directory containing `mod_name` package.  Typically that will be the
        directory containing the e.g. 'nibabel' directory.
    cmd : str
        Python command to execute
    script_dir : None or str, optional
        script directory to prepend to PATH
    print_location : bool, optional
        Whether to print the location of the imported `mod_name`

    Returns
    -------
    stdout : str
        stdout as str
    stderr : str
        stderr as str
    """
    if script_dir is None:
        paths_add = ''
    else:
        if not HAVE_PUTENV:
            raise RuntimeError('We cannot set environment variables')
        paths_add = '\nos.environ[\'PATH\'] = r\'"{script_dir}"\' + os.path.pathsep + os.environ[\'PATH\']\nPYTHONPATH = os.environ.get(\'PYTHONPATH\')\nif PYTHONPATH is None:\n    os.environ[\'PYTHONPATH\'] = r\'"{pkg_path}"\'\nelse:\n    os.environ[\'PYTHONPATH\'] = r\'"{pkg_path}"\' + os.path.pathsep + PYTHONPATH\n'.format(**locals())
    if print_location:
        p_loc = f'print({mod_name}.__file__);'
    else:
        p_loc = ''
    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    try:
        os.chdir(tmpdir)
        with open('script.py', 'wt') as fobj:
            fobj.write('\nimport os\nimport sys\nsys.path.insert(0, r"{pkg_path}")\n{paths_add}\nimport {mod_name}\n{p_loc}\n{cmd}'.format(**locals()))
        res = back_tick(f'{PYTHON} script.py', ret_err=True)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir)
    return res