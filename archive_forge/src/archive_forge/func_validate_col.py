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
def validate_col(self, itemsize=None):
    """validate this column: return the compared against itemsize"""
    if _ensure_decoded(self.kind) == 'string':
        c = self.col
        if c is not None:
            if itemsize is None:
                itemsize = self.itemsize
            if c.itemsize < itemsize:
                raise ValueError(f'Trying to store a string with len [{itemsize}] in [{self.cname}] column but\nthis column has a limit of [{c.itemsize}]!\nConsider using min_itemsize to preset the sizes on these columns')
            return c.itemsize
    return None