from __future__ import annotations
from typing import (
import warnings
from pandas._libs import lib
from pandas._libs.json import ujson_loads
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas import DataFrame
import pandas.core.common as com
from pandas.tseries.frequencies import to_offset
def convert_pandas_type_to_json_field(arr) -> dict[str, JSONSerializable]:
    dtype = arr.dtype
    name: JSONSerializable
    if arr.name is None:
        name = 'values'
    else:
        name = arr.name
    field: dict[str, JSONSerializable] = {'name': name, 'type': as_json_table_type(dtype)}
    if isinstance(dtype, CategoricalDtype):
        cats = dtype.categories
        ordered = dtype.ordered
        field['constraints'] = {'enum': list(cats)}
        field['ordered'] = ordered
    elif isinstance(dtype, PeriodDtype):
        field['freq'] = dtype.freq.freqstr
    elif isinstance(dtype, DatetimeTZDtype):
        if timezones.is_utc(dtype.tz):
            field['tz'] = 'UTC'
        else:
            field['tz'] = dtype.tz.zone
    elif isinstance(dtype, ExtensionDtype):
        field['extDtype'] = dtype.name
    return field