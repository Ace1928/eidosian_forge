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
def op_on_upcast_args(x, y):
    '\n        Return %s(self, to_affine_scalar(y)) if y can be upcast\n        through to_affine_scalar.  Otherwise returns NotImplemented.\n        ' % func.__name__
    try:
        y_with_uncert = to_affine_scalar(y)
    except NotUpcast:
        return NotImplemented
    else:
        return func(x, y_with_uncert)