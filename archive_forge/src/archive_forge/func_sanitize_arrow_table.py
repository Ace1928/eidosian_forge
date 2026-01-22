from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import json
import itertools
import re
import sys
import traceback
import warnings
from typing import (
from types import ModuleType
import jsonschema
import pandas as pd
import numpy as np
from pandas.api.types import infer_dtype
from altair.utils.schemapi import SchemaBase
from altair.utils._dfi_types import Column, DtypeKind, DataFrame as DfiDataFrame
from typing import Literal, Protocol, TYPE_CHECKING, runtime_checkable
def sanitize_arrow_table(pa_table):
    """Sanitize arrow table for JSON serialization"""
    import pyarrow as pa
    import pyarrow.compute as pc
    arrays = []
    schema = pa_table.schema
    for name in schema.names:
        array = pa_table[name]
        dtype_name = str(schema.field(name).type)
        if dtype_name.startswith('timestamp') or dtype_name.startswith('date'):
            arrays.append(pc.strftime(array))
        elif dtype_name.startswith('duration'):
            raise ValueError('Field "{col_name}" has type "{dtype}" which is not supported by Altair. Please convert to either a timestamp or a numerical value.'.format(col_name=name, dtype=dtype_name))
        else:
            arrays.append(array)
    return pa.Table.from_arrays(arrays, names=schema.names)