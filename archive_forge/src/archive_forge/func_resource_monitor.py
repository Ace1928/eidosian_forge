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
@resource_monitor.setter
def resource_monitor(self, value):
    if isinstance(value, (str, bytes)):
        value = str2bool(value.lower())
    if value is False:
        self._resource_monitor = False
    elif value is True:
        if not self._resource_monitor:
            self._resource_monitor = False
            try:
                import psutil
                self._resource_monitor = LooseVersion(psutil.__version__) >= LooseVersion('5.0')
            except ImportError:
                pass
            finally:
                if not self._resource_monitor:
                    warn('Could not enable the resource monitor: psutil>=5.0 could not be imported.')
                self._config.set('monitoring', 'enabled', ('%s' % self._resource_monitor).lower())