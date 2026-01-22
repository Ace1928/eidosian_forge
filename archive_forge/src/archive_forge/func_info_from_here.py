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
def info_from_here(mod_name):
    """Run info context checks starting in working directory

    Runs checks from current working directory, installing temporary
    installations into a new temporary directory

    Parameters
    ----------
    mod_name : str
       package name that will be installed, and tested
    """
    repo_path = os.path.abspath(os.getcwd())
    install_path = tempfile.mkdtemp()
    try:
        contexts_print_info(mod_name, repo_path, install_path)
    finally:
        shutil.rmtree(install_path)