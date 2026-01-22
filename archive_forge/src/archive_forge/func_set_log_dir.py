import os
import sys
import errno
import atexit
from warnings import warn
from looseversion import LooseVersion
import configparser
import numpy as np
from simplejson import load, dump
from .misc import str2bool
from filelock import SoftFileLock
def set_log_dir(self, log_dir):
    """Sets logging directory

        This should be the first thing that is done before any nipype class
        with logging is imported.
        """
    self._config.set('logging', 'log_directory', log_dir)