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
def missing_from(path0, path1, filter=None):
    """Return filenames present in `path0` but not in `path1`

    Parameters
    ----------
    path0 : str
        path which contains all files of interest
    path1 : str
        path which should contain all files of interest
    filter : None or str or regexp, optional
        A successful result from ``filter.search(fname)`` means the file is of
        interest.  None means all files are of interest

    Returns
    -------
    path1_missing : list
        list of all files missing from `path1` that are in `path0` at the same
        relative path.
    """
    if not filter is None:
        filter = re.compile(filter)
    uninstalled = []
    for dirpath, dirnames, filenames in os.walk(path0):
        out_dirpath = dirpath.replace(path0, path1)
        for fname in filenames:
            if not filter is None and filter.search(fname) is None:
                continue
            equiv_fname = os.path.join(out_dirpath, fname)
            if not os.path.isfile(equiv_fname):
                uninstalled.append(pjoin(dirpath, fname))
    return uninstalled