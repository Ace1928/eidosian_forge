import abc
import collections.abc
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import sexp
from . import conversion
import rpy2.rlike.container as rlc
import datetime
import copy
import itertools
import math
import os
import jinja2  # type: ignore
import time
import tzlocal
from time import struct_time, mktime
import typing
import warnings
from rpy2.rinterface import (Sexp, ListSexpVector, StrSexpVector,
@names.setter
def names(self, value) -> None:
    """ Set list of name vectors
        (like the R function 'dimnames' does)."""
    value = conversion.get_conversion().rpy2py(value)
    res = self._dimnames_set(self, value)
    self.__sexp__ = res.__sexp__