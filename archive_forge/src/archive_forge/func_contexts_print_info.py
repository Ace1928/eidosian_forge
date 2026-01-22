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
def contexts_print_info(mod_name, repo_path, install_path):
    """Print result of get_info from different installation routes

    Runs installation from:

    * git archive zip file
    * with setup.py install from repository directory
    * just running code from repository directory

    and prints out result of get_info in each case.  There will be many files
    written into `install_path` that you may want to clean up somehow.

    Parameters
    ----------
    mod_name : str
       package name that will be installed, and tested
    repo_path : str
       path to location of git repository
    install_path : str
       path into which to install temporary installations
    """
    site_pkgs_path = os.path.join(install_path, PY_LIB_SDIR)
    pwd = os.path.abspath(os.getcwd())
    out_fname = pjoin(install_path, 'test.zip')
    try:
        os.chdir(repo_path)
        back_tick(f'git archive --format zip -o {out_fname} HEAD')
    finally:
        os.chdir(pwd)
    install_from_zip(out_fname, install_path, None)
    cmd_str = f'print({mod_name}.get_info())'
    print(run_mod_cmd(mod_name, site_pkgs_path, cmd_str)[0])
    install_from_to(repo_path, install_path, PY_LIB_SDIR)
    print(run_mod_cmd(mod_name, site_pkgs_path, cmd_str)[0])
    print(run_mod_cmd(mod_name, repo_path, cmd_str)[0])