from __future__ import print_function, division, absolute_import
from .dispatch import dispatch
from .coretypes import (
from .predicates import isdimension
from .util import dshape
from datetime import date, time, datetime
import numpy as np
@dispatch(DataShape, DataShape)
def issubschema(a, b):
    if a == b:
        return True
    return None