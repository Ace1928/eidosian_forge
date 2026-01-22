from __future__ import annotations
import sys
import typing
from abc import ABC, abstractmethod
from datetime import MAXYEAR, MINYEAR, datetime, timedelta
from types import MethodType
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from ._core.dates import datetime_to_num, num_to_datetime
from .breaks import (
from .labels import (
from .utils import identity
def minor_breaks(self, major: FloatArrayLike, limits: Optional[TupleFloat2]=None, n: Optional[int]=None) -> NDArrayFloat:
    n = int(self.base) - 2 if n is None else n
    return super().minor_breaks(major, limits, n)