import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def still_active(self):
    """
        Return whether this finalizer is still waiting to invoke callback
        """
    return self._key in _finalizer_registry