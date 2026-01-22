import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def set_break(self, filename, lineno, temporary=False, cond=None, funcname=None):
    """Set a new breakpoint for filename:lineno.

        If lineno doesn't exist for the filename, return an error message.
        The filename should be in canonical form.
        """
    filename = self.canonic(filename)
    import linecache
    line = linecache.getline(filename, lineno)
    if not line:
        return 'Line %s:%d does not exist' % (filename, lineno)
    self._add_to_breaks(filename, lineno)
    bp = Breakpoint(filename, lineno, temporary, cond, funcname)
    return None