from __future__ import absolute_import
import re
from collections import namedtuple
from ..exceptions import LocationParseError
from ..packages import six
def ensure_type(x):
    return x if x is None else ensure_func(x)