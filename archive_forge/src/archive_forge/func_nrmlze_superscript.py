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
def nrmlze_superscript(number_str):
    """
    Return a string with superscript digits transformed into regular digits.

    Non-superscript digits are not changed before the conversion. Thus, the
    string can also contain regular digits.

    ValueError is raised if the conversion cannot be done.

    number_str -- string to be converted (of type str, but also possibly, for 
    Python 2, unicode, which allows this string to contain superscript digits).
    """
    return int(str(number_str).translate(FROM_SUPERSCRIPT))