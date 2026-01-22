from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def searchupwards(start, files=[], dirs=[]):
    """
    Walk upwards from start, looking for a directory containing
    all files and directories given as arguments::
    >>> searchupwards('.', ['foo.txt'], ['bar', 'bam'])

    If not found, return None
    """
    start = os.path.abspath(start)
    parents = start.split(os.sep)
    exists = os.path.exists
    join = os.sep.join
    isdir = os.path.isdir
    while len(parents):
        candidate = join(parents) + os.sep
        allpresent = 1
        for f in files:
            if not exists(f'{candidate}{f}'):
                allpresent = 0
                break
        if allpresent:
            for d in dirs:
                if not isdir(f'{candidate}{d}'):
                    allpresent = 0
                    break
        if allpresent:
            return candidate
        parents.pop(-1)
    return None