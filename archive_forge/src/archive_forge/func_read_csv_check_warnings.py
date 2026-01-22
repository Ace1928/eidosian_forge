from __future__ import annotations
import os
import pytest
from pandas.compat._optional import VERSIONS
from pandas import (
import pandas._testing as tm
def read_csv_check_warnings(self, warn_type: type[Warning], warn_msg: str, *args, raise_on_extra_warnings=True, check_stacklevel: bool=True, **kwargs):
    kwargs = self.update_kwargs(kwargs)
    with tm.assert_produces_warning(warn_type, match=warn_msg, raise_on_extra_warnings=raise_on_extra_warnings, check_stacklevel=check_stacklevel):
        return read_csv(*args, **kwargs)