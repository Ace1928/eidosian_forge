from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
def ignores_leakcheck(func):
    """
    Ignore the given object during leakchecks.

    Can be applied to a method, in which case the method will run, but
    will not be subject to leak checks.

    If applied to a class, the entire class will be skipped during leakchecks. This
    is intended to be used for classes that are very slow and cause problems such as
    test timeouts; typically it will be used for classes that are subclasses of a base
    class and specify variants of behaviour (such as pool sizes).
    """
    func.ignore_leakcheck = True
    return func