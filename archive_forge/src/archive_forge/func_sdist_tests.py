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
def sdist_tests(mod_name, repo_path=None, label='fast', doctests=True):
    """Make sdist zip, install from it, and run tests"""
    if repo_path is None:
        repo_path = abspath(os.getcwd())
    install_path = tempfile.mkdtemp()
    try:
        zip_fname = make_dist(repo_path, install_path, 'sdist --formats=zip', '*.zip')
        pf = get_sdist_finder(mod_name)
        install_from_zip(zip_fname, install_path, pf, PY_LIB_SDIR, 'bin')
        site_pkgs_path = pjoin(install_path, PY_LIB_SDIR)
        script_path = pjoin(install_path, 'bin')
        cmd = f"{mod_name}.test(label='{label}', doctests={doctests})"
        stdout, stderr = run_mod_cmd(mod_name, site_pkgs_path, cmd, script_path)
    finally:
        shutil.rmtree(install_path)
    print(stdout)
    print(stderr)