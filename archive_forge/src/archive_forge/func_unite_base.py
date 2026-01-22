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
def unite_base(dshapes):
    """ Performs lowest common dshape and also null aware

    >>> unite_base([float64, float64, int64])
    dshape("3 * float64")

    >>> unite_base([int32, int64, null])
    dshape("3 * ?int64")
    """
    dshapes = [unpack(ds) for ds in dshapes]
    bynull = groupby(isnull, dshapes)
    try:
        good_dshapes = bynull[False]
    except KeyError:
        return len(dshapes) * null
    if all((isinstance(ds, Unit) for ds in good_dshapes)):
        base = lowest_common_dshape(good_dshapes)
    elif (all((isinstance(ds, Record) for ds in good_dshapes)) and ds.names == dshapes[0].names for ds in good_dshapes):
        names = good_dshapes[0].names
        base = Record([[name, unite_base([ds.dict.get(name, null) for ds in good_dshapes]).subshape[0]] for name in names])
    if base:
        if bynull.get(True):
            base = Option(base)
        return len(dshapes) * base