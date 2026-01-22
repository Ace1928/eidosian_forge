from abc import abstractmethod
import math
import operator
import re
import datetime
from calendar import isleap
from decimal import Decimal, Context
from typing import cast, Any, Callable, Dict, Optional, Tuple, Union
from ..helpers import MONTH_DAYS_LEAP, MONTH_DAYS, DAYS_IN_4Y, \
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
@classmethod
def fromtimedelta(cls, td: datetime.timedelta) -> 'DayTimeDuration':
    return cls(seconds=Decimal('{}.{:06}'.format(td.days * 86400 + td.seconds, td.microseconds)))