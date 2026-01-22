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
def bdist_egg_tests(mod_name, repo_path=None, label='fast', doctests=True):
    """Make bdist_egg, unzip it, and run tests from result

    We've got a problem here, because the egg does not contain the scripts, and
    so, if we are testing the scripts with ``mod.test()``, we won't pick up the
    scripts from the repository we are testing.

    So, you might need to add a label to the script tests, and use the `label`
    parameter to indicate these should be skipped. As in:

        bdist_egg_tests('nibabel', None, label='not script_test')
    """
    if repo_path is None:
        repo_path = abspath(os.getcwd())
    install_path = tempfile.mkdtemp()
    scripts_path = pjoin(install_path, 'bin')
    try:
        zip_fname = make_dist(repo_path, install_path, 'bdist_egg', '*.egg')
        zip_extract_all(zip_fname, install_path)
        cmd = f"{mod_name}.test(label='{label}', doctests={doctests})"
        stdout, stderr = run_mod_cmd(mod_name, install_path, cmd, scripts_path)
    finally:
        shutil.rmtree(install_path)
    print(stdout)
    print(stderr)