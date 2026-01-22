import collections
import os
import re
import sys
import functools
import itertools
def python_build():
    """ Returns a tuple (buildno, builddate) stating the Python
        build number and date as strings.

    """
    return _sys_version()[4:6]