import configparser
import glob
import os
import sys
from os.path import join as pjoin
from packaging.version import Version
from .environment import get_nipy_system_dir, get_nipy_user_dir
def find_data_dir(root_dirs, *names):
    """Find relative path given path prefixes to search

    We raise a DataError if we can't find the relative path

    Parameters
    ----------
    root_dirs : sequence of strings
       sequence of paths in which to search for data directory
    *names : sequence of strings
       sequence of strings naming directory to find. The name to search
       for is given by ``os.path.join(*names)``

    Returns
    -------
    data_dir : str
       full path (root path added to `*names` above)

    """
    ds_relative = pjoin(*names)
    for path in root_dirs:
        pth = pjoin(path, ds_relative)
        if os.path.isdir(pth):
            return pth
    raise DataError(f'Could not find datasource "{ds_relative}" in data path "{os.path.pathsep.join(root_dirs)}"')