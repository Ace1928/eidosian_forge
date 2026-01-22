import os
import sys
import tempfile
import operator
import functools
import itertools
import re
import contextlib
import pickle
import textwrap
import builtins
import pkg_resources
from distutils.errors import DistutilsError
from pkg_resources import working_set
@contextlib.contextmanager
def save_pkg_resources_state():
    saved = pkg_resources.__getstate__()
    try:
        yield saved
    finally:
        pkg_resources.__setstate__(saved)