from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
def ufloat_obsolete(representation, tag=None):
    """
    Legacy version of ufloat(). Will eventually be removed.

    representation -- either a (nominal_value, std_dev) tuple, or a
    string representation of a number with uncertainty, in a format
    recognized by ufloat_fromstr().
    """
    if isinstance(representation, tuple):
        return ufloat(representation[0], representation[1], tag)
    else:
        return ufloat_fromstr(representation, tag)