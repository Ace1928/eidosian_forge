from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Union, overload
import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis.errors import InvalidArgument
import xarray as xr
from xarray.core.types import T_DuckArray
def supported_dtypes() -> st.SearchStrategy[np.dtype]:
    """
    Generates only those numpy dtypes which xarray can handle.

    Use instead of hypothesis.extra.numpy.scalar_dtypes in order to exclude weirder dtypes such as unicode, byte_string, array, or nested dtypes.
    Also excludes datetimes, which dodges bugs with pandas non-nanosecond datetime overflows.

    Requires the hypothesis package to be installed.

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    return npst.integer_dtypes() | npst.unsigned_integer_dtypes() | npst.floating_dtypes() | npst.complex_number_dtypes()