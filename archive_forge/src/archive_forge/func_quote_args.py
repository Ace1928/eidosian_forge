import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def quote_args(args):
    """Quote list of arguments.

    .. deprecated:: 1.22.
    """
    import warnings
    warnings.warn('"quote_args" is deprecated.', DeprecationWarning, stacklevel=2)
    args = list(args)
    for i in range(len(args)):
        a = args[i]
        if ' ' in a and a[0] not in '"\'':
            args[i] = '"%s"' % a
    return args