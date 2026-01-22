import collections
import os
import re
import sys
import functools
import itertools
def python_branch():
    """ Returns a string identifying the Python implementation
        branch.

        For CPython this is the SCM branch from which the
        Python binary was built.

        If not available, an empty string is returned.

    """
    return _sys_version()[2]