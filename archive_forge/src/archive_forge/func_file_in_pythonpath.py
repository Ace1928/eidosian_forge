import fnmatch
import logging
import os
import sys
import types
from typing import Optional, Set
def file_in_pythonpath(filepath) -> bool:
    """Test whether a filepath is in the same folder of a path specified in the PYTHONPATH env variable.


    Parameters
    ----------
    filepath : str
        An absolute file path.

    Returns
    -------
    boolean
        True if contained in PYTHONPATH, False otherwise. False if PYTHONPATH is not defined or empty.

    """
    pythonpath = os.environ.get('PYTHONPATH', '')
    if len(pythonpath) == 0:
        return False
    absolute_paths = [os.path.abspath(path) for path in pythonpath.split(os.pathsep)]
    return any((file_is_in_folder_glob(os.path.normpath(filepath), path) for path in absolute_paths))