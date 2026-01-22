import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
@daemon.setter
def daemon(self, daemonic):
    """
        Set whether process is a daemon
        """
    assert self._popen is None, 'process has already started'
    self._config['daemon'] = daemonic