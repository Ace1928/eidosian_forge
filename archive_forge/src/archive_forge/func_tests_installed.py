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
def tests_installed(mod_name, source_path=None):
    """Install from `source_path` into temporary directory; run tests

    Parameters
    ----------
    mod_name : str
        name of module - e.g. 'nibabel'
    source_path : None or str
        Path from which to install.  If None, defaults to working directory
    """
    if source_path is None:
        source_path = os.path.abspath(os.getcwd())
    install_path = tempfile.mkdtemp()
    site_pkgs_path = pjoin(install_path, PY_LIB_SDIR)
    scripts_path = pjoin(install_path, 'bin')
    try:
        install_from_to(source_path, install_path, PY_LIB_SDIR, 'bin')
        stdout, stderr = run_mod_cmd(mod_name, site_pkgs_path, mod_name + '.test()', scripts_path)
    finally:
        shutil.rmtree(install_path)
    print(stdout)
    print(stderr)