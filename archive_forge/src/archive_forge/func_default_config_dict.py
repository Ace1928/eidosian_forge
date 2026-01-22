import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def default_config_dict(name=None, parent_name=None, local_path=None):
    """Return a configuration dictionary for usage in
    configuration() function defined in file setup_<name>.py.
    """
    import warnings
    warnings.warn('Use Configuration(%r,%r,top_path=%r) instead of deprecated default_config_dict(%r,%r,%r)' % (name, parent_name, local_path, name, parent_name, local_path), stacklevel=2)
    c = Configuration(name, parent_name, local_path)
    return c.todict()