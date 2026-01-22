import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def push_caller(self, caller):
    """Push a ``caller`` callable onto the callstack for
        this :class:`.Context`."""
    self.caller_stack.append(caller)