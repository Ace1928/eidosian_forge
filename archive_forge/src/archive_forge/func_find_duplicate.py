import warnings
from collections import Counter
from contextlib import nullcontext
from .._utils import set_module
from . import numeric as sb
from . import numerictypes as nt
from numpy.compat import os_fspath
from .arrayprint import _get_legacy_print_mode
def find_duplicate(list):
    """Find duplication in a list, return a list of duplicated elements"""
    return [item for item, counts in Counter(list).items() if counts > 1]