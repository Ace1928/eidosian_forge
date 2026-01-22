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
def validate_metadata(self, handler: AppendableTable) -> None:
    """validate that kind=category does not change the categories"""
    if self.meta == 'category':
        new_metadata = self.metadata
        cur_metadata = handler.read_metadata(self.cname)
        if new_metadata is not None and cur_metadata is not None and (not array_equivalent(new_metadata, cur_metadata, strict_nan=True, dtype_equal=True)):
            raise ValueError('cannot append a categorical with different categories to the existing')