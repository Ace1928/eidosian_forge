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
def ufloat(nominal_value, std_dev=None, tag=None):
    """
    Return a new random variable (Variable object).

    The only non-obsolete use is:

    - ufloat(nominal_value, std_dev),
    - ufloat(nominal_value, std_dev, tag=...).

    Other input parameters are temporarily supported:

    - ufloat((nominal_value, std_dev)),
    - ufloat((nominal_value, std_dev), tag),
    - ufloat(str_representation),
    - ufloat(str_representation, tag).

    Valid string representations str_representation are listed in
    the documentation for ufloat_fromstr().

    nominal_value -- nominal value of the random variable. It is more
    meaningful to use a value close to the central value or to the
    mean. This value is propagated by mathematical operations as if it
    was a float.

    std_dev -- standard deviation of the random variable. The standard
    deviation must be convertible to a positive float, or be NaN.

    tag -- optional string tag for the variable.  Variables don't have
    to have distinct tags.  Tags are useful for tracing what values
    (and errors) enter in a given result (through the
    error_components() method).
    """
    try:
        return Variable(nominal_value, std_dev, tag=tag)
    except (TypeError, ValueError):
        if tag is not None:
            tag_arg = tag
        else:
            tag_arg = std_dev
        try:
            final_ufloat = ufloat_obsolete(nominal_value, tag_arg)
        except:
            raise
        else:
            deprecation('either use ufloat(nominal_value, std_dev), ufloat(nominal_value, std_dev, tag), or the ufloat_fromstr() function, for string representations.')
            return final_ufloat