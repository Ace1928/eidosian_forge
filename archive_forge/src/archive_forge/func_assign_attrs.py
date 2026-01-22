from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import _raise_if_any_duplicate_dimensions
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import is_chunked_array
def assign_attrs(self, *args: Any, **kwargs: Any) -> Self:
    """Assign new attrs to this object.

        Returns a new object equivalent to ``self.attrs.update(*args, **kwargs)``.

        Parameters
        ----------
        *args
            positional arguments passed into ``attrs.update``.
        **kwargs
            keyword arguments passed into ``attrs.update``.

        Examples
        --------
        >>> dataset = xr.Dataset({"temperature": [25, 30, 27]})
        >>> dataset
        <xarray.Dataset> Size: 24B
        Dimensions:      (temperature: 3)
        Coordinates:
          * temperature  (temperature) int64 24B 25 30 27
        Data variables:
            *empty*

        >>> new_dataset = dataset.assign_attrs(
        ...     units="Celsius", description="Temperature data"
        ... )
        >>> new_dataset
        <xarray.Dataset> Size: 24B
        Dimensions:      (temperature: 3)
        Coordinates:
          * temperature  (temperature) int64 24B 25 30 27
        Data variables:
            *empty*
        Attributes:
            units:        Celsius
            description:  Temperature data

        # Attributes of the new dataset

        >>> new_dataset.attrs
        {'units': 'Celsius', 'description': 'Temperature data'}

        Returns
        -------
        assigned : same type as caller
            A new object with the new attrs in addition to the existing data.

        See Also
        --------
        Dataset.assign
        """
    out = self.copy(deep=False)
    out.attrs.update(*args, **kwargs)
    return out