import collections
import os
import re
import sys
import functools
import itertools
def python_compiler():
    """ Returns a string identifying the compiler used for compiling
        Python.

    """
    return _sys_version()[6]