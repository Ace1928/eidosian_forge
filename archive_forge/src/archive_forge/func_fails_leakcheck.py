from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
def fails_leakcheck(func):
    """
    Mark that the function is known to leak.
    """
    func.fails_leakcheck = True
    if SKIP_FAILING_LEAKCHECKS:
        func = unittest.skip('Skipping known failures')(func)
    return func