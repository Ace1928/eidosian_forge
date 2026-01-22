from __future__ import annotations
import operator
import re
from re import Pattern
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import isna
def should_use_regex(regex: bool, to_replace: Any) -> bool:
    """
    Decide whether to treat `to_replace` as a regular expression.
    """
    if is_re(to_replace):
        regex = True
    regex = regex and is_re_compilable(to_replace)
    regex = regex and re.compile(to_replace).pattern != ''
    return regex