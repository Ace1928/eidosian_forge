import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def is_skipped_module(self, module_name):
    """Return True if module_name matches any skip pattern."""
    if module_name is None:
        return False
    for pattern in self.skip:
        if fnmatch.fnmatch(module_name, pattern):
            return True
    return False