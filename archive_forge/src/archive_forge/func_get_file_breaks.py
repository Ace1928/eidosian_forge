import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def get_file_breaks(self, filename):
    """Return all lines with breakpoints for filename.

        If no breakpoints are set, return an empty list.
        """
    filename = self.canonic(filename)
    if filename in self.breaks:
        return self.breaks[filename]
    else:
        return []