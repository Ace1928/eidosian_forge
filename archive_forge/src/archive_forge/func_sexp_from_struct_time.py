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
def sexp_from_struct_time(seq):

    def f(seq):
        return [IntVector([x.tm_year for x in seq]), IntVector([x.tm_mon for x in seq]), IntVector([x.tm_mday for x in seq]), IntVector([x.tm_hour for x in seq]), IntVector([x.tm_min for x in seq]), IntVector([x.tm_sec for x in seq])]
    return POSIXct._sexp_from_seq(seq, lambda elt: elt.tm_zone, f)