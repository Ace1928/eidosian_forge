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
def iter_labels(self):
    """ Iterate the over the labels, that is iterate over
        the items returning associated label for each item """
    levels = self.levels
    for x in conversion.noconversion(self):
        yield (rinterface.NA_Character if x is rinterface.NA_Integer else levels[x - 1])