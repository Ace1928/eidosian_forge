import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def notifyAll(self):
    """Wake up all threads waiting on this condition.

        This method is deprecated, use notify_all() instead.

        """
    import warnings
    warnings.warn('notifyAll() is deprecated, use notify_all() instead', DeprecationWarning, stacklevel=2)
    self.notify_all()