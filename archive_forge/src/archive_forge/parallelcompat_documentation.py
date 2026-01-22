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

        Store chunked arrays in array-like objects, overwriting data in target.

        This stores chunked arrays into object that supports numpy-style setitem
        indexing (e.g. a Zarr Store). Allows storing values chunk by chunk so that it does not have to
        fill up memory. For best performance you likely want to align the block size of
        the storage target with the block size of your array.

        Used when writing to any registered xarray I/O backend.

        Parameters
        ----------
        sources: Array or collection of Arrays
        targets: array-like or collection of array-likes
            These should support setitem syntax ``target[10:20] = ...``.
            If sources is a single item, targets must be a single item; if sources is a
            collection of arrays, targets must be a matching collection.
        kwargs:
            Parameters passed to compute/persist (only used if compute=True)

        See Also
        --------
        dask.array.store
        cubed.store
        