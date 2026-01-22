from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
def fails_strict_leakcheck(func):
    """
        Decorator for a function that is known to fail when running
        strict (``sys.getobjects()``) leakchecks.

        This type of leakcheck finds all objects, even those, such as
        strings, which are not tracked by the garbage collector.
        """
    return func