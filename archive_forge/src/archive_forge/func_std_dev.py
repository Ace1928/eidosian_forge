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
@std_dev.setter
def std_dev(self, std_dev):
    if std_dev < 0 and (not isinfinite(std_dev)):
        raise NegativeStdDev('The standard deviation cannot be negative')
    self._std_dev = CallableStdDev(std_dev)