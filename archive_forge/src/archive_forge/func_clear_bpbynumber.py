import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def clear_bpbynumber(self, arg):
    """Delete a breakpoint by its index in Breakpoint.bpbynumber.

        If arg is invalid, return an error message.
        """
    try:
        bp = self.get_bpbynumber(arg)
    except ValueError as err:
        return str(err)
    bp.deleteMe()
    self._prune_breaks(bp.file, bp.line)
    return None