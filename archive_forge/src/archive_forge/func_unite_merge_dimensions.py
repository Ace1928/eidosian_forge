from __future__ import print_function, division, absolute_import
from collections import OrderedDict
from datetime import datetime, date, time, timedelta
from itertools import chain
import re
from textwrap import dedent
from types import MappingProxyType
from warnings import warn
from dateutil.parser import parse as dateparse
import numpy as np
from .dispatch import dispatch
from .coretypes import (int32, int64, float64, bool_, complex128, datetime_,
from .predicates import isdimension, isrecord
from .internal_utils import _toposort, groupby
from .util import subclasses
def unite_merge_dimensions(dshapes, unite=unite_identical):
    """

    >>> unite_merge_dimensions([10 * string, 10 * string])
    dshape("2 * 10 * string")

    >>> unite_merge_dimensions([10 * string, 20 * string])
    dshape("2 * var * string")
    """
    n = len(dshapes)
    if all((isinstance(ds, DataShape) and isdimension(ds[0]) for ds in dshapes)):
        dims = [ds[0] for ds in dshapes]
        base = unite([ds.subshape[0] for ds in dshapes])
        if base:
            if len(set(dims)) == 1:
                return n * (dims[0] * base.subshape[0])
            else:
                return n * (var * base.subshape[0])