import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
@property
def sentinel(self):
    """
        Return a file descriptor (Unix) or handle (Windows) suitable for
        waiting for process termination.
        """
    self._check_closed()
    try:
        return self._sentinel
    except AttributeError:
        raise ValueError('process not started') from None