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
def save_modules():
    """
    Context in which imported modules are saved.

    Translates exceptions internal to the context into the equivalent exception
    outside the context.
    """
    saved = sys.modules.copy()
    with ExceptionSaver() as saved_exc:
        yield saved
    sys.modules.update(saved)
    del_modules = (mod_name for mod_name in sys.modules if mod_name not in saved and (not mod_name.startswith('encodings.')))
    _clear_modules(del_modules)
    saved_exc.resume()