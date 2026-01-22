from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Literal, Union
import numpy as np
import pandas as pd
from xarray.coding import strings, times, variables
from xarray.coding.variables import SerializationWarning, pop_to
from xarray.core import indexing
from xarray.core.common import (
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import IndexVariable, Variable
from xarray.namedarray.utils import is_duck_dask_array
def encode_cf_variable(var: Variable, needs_copy: bool=True, name: T_Name=None) -> Variable:
    """
    Converts a Variable into a Variable which follows some
    of the CF conventions:

        - Nans are masked using _FillValue (or the deprecated missing_value)
        - Rescaling via: scale_factor and add_offset
        - datetimes are converted to the CF 'units since time' format
        - dtype encodings are enforced.

    Parameters
    ----------
    var : Variable
        A variable holding un-encoded data.

    Returns
    -------
    out : Variable
        A variable which has been encoded as described above.
    """
    ensure_not_multiindex(var, name=name)
    for coder in [times.CFDatetimeCoder(), times.CFTimedeltaCoder(), variables.CFScaleOffsetCoder(), variables.CFMaskCoder(), variables.UnsignedIntegerCoder(), variables.NativeEnumCoder(), variables.NonStringCoder(), variables.DefaultFillvalueCoder(), variables.BooleanCoder()]:
        var = coder.encode(var, name=name)
    var = ensure_dtype_not_object(var, name=name)
    for attr_name in CF_RELATED_DATA:
        pop_to(var.encoding, var.attrs, attr_name)
    return var