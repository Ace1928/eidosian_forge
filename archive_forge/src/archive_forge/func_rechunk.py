from __future__ import annotations
import functools
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar
import numpy as np
from xarray.core.utils import emit_user_level_warning
from xarray.namedarray.pycompat import is_chunked_array
def rechunk(self, data: T_ChunkedArray, chunks: _NormalizedChunks | tuple[int, ...] | _Chunks, **kwargs: Any) -> Any:
    """
        Changes the chunking pattern of the given array.

        Called when the .chunk method is called on an xarray object that is already chunked.

        Parameters
        ----------
        data : dask array
            Array to be rechunked.
        chunks :  int, tuple, dict or str, optional
            The new block dimensions to create. -1 indicates the full size of the
            corresponding dimension. Default is "auto" which automatically
            determines chunk sizes.

        Returns
        -------
        chunked array

        See Also
        --------
        dask.array.Array.rechunk
        cubed.Array.rechunk
        """
    return data.rechunk(chunks, **kwargs)