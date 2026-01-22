import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def setDaemon(self, daemonic):
    """Set whether this thread is a daemon.

        This method is deprecated, use the .daemon property instead.

        """
    import warnings
    warnings.warn('setDaemon() is deprecated, set the daemon attribute instead', DeprecationWarning, stacklevel=2)
    self.daemon = daemonic