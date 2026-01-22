from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
def queryables(self) -> dict[str, Any]:
    """return a dict of the kinds allowable columns for this object"""
    axis_names = {0: 'index', 1: 'columns'}
    d1 = [(a.cname, a) for a in self.index_axes]
    d2 = [(axis_names[axis], None) for axis, values in self.non_index_axes]
    d3 = [(v.cname, v) for v in self.values_axes if v.name in set(self.data_columns)]
    return dict(d1 + d2 + d3)