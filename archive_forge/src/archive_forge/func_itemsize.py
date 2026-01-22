from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
import numpy as np
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.compat import pa_version_under7p0
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.base import (
from pandas.core.dtypes.dtypes import CategoricalDtypeType
@cache_readonly
def itemsize(self) -> int:
    """Return the number of bytes in this dtype"""
    return self.numpy_dtype.itemsize