from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
def check_case_sensitive(self, name: str, schema: str | None) -> None:
    """
        Checks table name for issues with case-sensitivity.
        Method is called after data is inserted.
        """
    if not name.isdigit() and (not name.islower()):
        from sqlalchemy import inspect as sqlalchemy_inspect
        insp = sqlalchemy_inspect(self.con)
        table_names = insp.get_table_names(schema=schema or self.meta.schema)
        if name not in table_names:
            msg = f"The provided table name '{name}' is not found exactly as such in the database after writing the table, possibly due to case sensitivity issues. Consider using lower case table names."
            warnings.warn(msg, UserWarning, stacklevel=find_stack_level())