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
def lowest_common_dshape(dshapes):
    """ Find common shared dshape

    >>> lowest_common_dshape([int32, int64, float64])
    ctype("float64")

    >>> lowest_common_dshape([int32, int64])
    ctype("int64")

    >>> lowest_common_dshape([string, int64])
    ctype("string")
    """
    common = set.intersection(*[descendents(edges, ds) for ds in dshapes])
    if common and any((c in toposorted for c in common)):
        return min(common, key=toposorted.index)
    raise ValueError('Not all dshapes are known.  Extend edges.')