from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Union, overload
import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis.errors import InvalidArgument
import xarray as xr
from xarray.core.types import T_DuckArray
def dimension_names(*, min_dims: int=0, max_dims: int=3) -> st.SearchStrategy[list[Hashable]]:
    """
    Generates an arbitrary list of valid dimension names.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    min_dims
        Minimum number of dimensions in generated list.
    max_dims
        Maximum number of dimensions in generated list.
    """
    return st.lists(elements=names(), min_size=min_dims, max_size=max_dims, unique=True)