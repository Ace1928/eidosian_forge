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
@staticmethod
def sexp_from_datetime(seq):
    """ return a POSIXct vector from a sequence of
        datetime.datetime elements. """

    def f(seq):
        return [IntVector([x.year for x in seq]), IntVector([x.month for x in seq]), IntVector([x.day for x in seq]), IntVector([x.hour for x in seq]), IntVector([x.minute for x in seq]), IntVector([x.second for x in seq])]

    def get_tz(elt):
        return elt.tzinfo if elt.tzinfo else None
    return POSIXct._sexp_from_seq(seq, get_tz, f)