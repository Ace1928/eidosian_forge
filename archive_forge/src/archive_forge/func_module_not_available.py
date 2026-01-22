import os
import shutil
import sys
import tempfile
import unittest
from importlib import import_module
from decorator import decorator
from .ipunittest import ipdoctest, ipdocstring
def module_not_available(module):
    """Can module be imported?  Returns true if module does NOT import.

    This is used to make a decorator to skip tests that require module to be
    available, but delay the 'import numpy' to test execution time.
    """
    try:
        mod = import_module(module)
        mod_not_avail = False
    except ImportError:
        mod_not_avail = True
    return mod_not_avail