import math
import re
from collections import OrderedDict, deque
from collections.abc import Hashable as CollectionsHashable
from datetime import date, datetime, time, timedelta
from decimal import Decimal, DecimalException
from enum import Enum, IntEnum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import (
from uuid import UUID
from . import errors
from .datetime_parse import parse_date, parse_datetime, parse_duration, parse_time
from .typing import (
from .utils import almost_equal_floats, lenient_issubclass, sequence_like
def number_size_validator(v: 'Number', field: 'ModelField') -> 'Number':
    field_type: ConstrainedNumber = field.type_
    if field_type.gt is not None and (not v > field_type.gt):
        raise errors.NumberNotGtError(limit_value=field_type.gt)
    elif field_type.ge is not None and (not v >= field_type.ge):
        raise errors.NumberNotGeError(limit_value=field_type.ge)
    if field_type.lt is not None and (not v < field_type.lt):
        raise errors.NumberNotLtError(limit_value=field_type.lt)
    if field_type.le is not None and (not v <= field_type.le):
        raise errors.NumberNotLeError(limit_value=field_type.le)
    return v