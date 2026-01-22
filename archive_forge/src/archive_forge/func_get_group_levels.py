from __future__ import annotations
import collections
import functools
from typing import (
import numpy as np
from pandas._libs import (
import pandas._libs.groupby as libgroupby
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import (
@final
def get_group_levels(self) -> list[ArrayLike]:
    if len(self.groupings) == 1:
        return [self.groupings[0]._group_arraylike]
    name_list = []
    for ping, codes in zip(self.groupings, self.reconstructed_codes):
        codes = ensure_platform_int(codes)
        levels = ping._group_arraylike.take(codes)
        name_list.append(levels)
    return name_list