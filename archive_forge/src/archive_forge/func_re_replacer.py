from __future__ import annotations
import operator
import re
from re import Pattern
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import isna
def re_replacer(s):
    if is_re(rx) and isinstance(s, str):
        return rx.sub(value, s)
    else:
        return s