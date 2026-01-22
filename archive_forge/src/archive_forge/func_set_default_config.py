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
def set_default_config(self):
    """Read default settings template and set into config object"""
    default_cfg = DEFAULT_CONFIG_TPL.format(log_dir=os.path.expanduser('~'), crashdump_dir=self.cwd)
    try:
        self._config.read_string(default_cfg)
    except AttributeError:
        from io import StringIO
        self._config.readfp(StringIO(default_cfg))