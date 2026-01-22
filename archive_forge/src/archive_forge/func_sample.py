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
def sample(self: collections.abc.Sized, n: int, replace: bool=False, probabilities: typing.Optional[collections.abc.Sized]=None):
    """ Draw a random sample of size n from the vector.

        If 'replace' is True, the sampling is done with replacement.
        The optional argument 'probabilities' can indicate sampling
        probabilities."""
    assert isinstance(n, int)
    assert isinstance(replace, bool)
    if probabilities is not None:
        if len(probabilities) != len(self):
            raise ValueError('The sequence of probabilities must match the length of the vector.')
        if not isinstance(probabilities, rinterface.FloatSexpVector):
            probabilities = FloatVector(probabilities)
    res = _sample(self, IntVector((n,)), replace=BoolVector((replace,)), prob=probabilities)
    res = conversion.rpy2py(res)
    return res