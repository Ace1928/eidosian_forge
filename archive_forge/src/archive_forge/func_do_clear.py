import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def do_clear(self, arg):
    """Remove temporary breakpoint.

        Must implement in derived classes or get NotImplementedError.
        """
    raise NotImplementedError('subclass of bdb must implement do_clear()')