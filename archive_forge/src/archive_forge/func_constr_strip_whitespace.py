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
def constr_strip_whitespace(v: 'StrBytes', field: 'ModelField', config: 'BaseConfig') -> 'StrBytes':
    strip_whitespace = field.type_.strip_whitespace or config.anystr_strip_whitespace
    if strip_whitespace:
        v = v.strip()
    return v