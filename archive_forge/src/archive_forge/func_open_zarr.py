from __future__ import annotations
import json
import os
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding, conventions
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import integer_types
def open_zarr(store, group=None, synchronizer=None, chunks='auto', decode_cf=True, mask_and_scale=True, decode_times=True, concat_characters=True, decode_coords=True, drop_variables=None, consolidated=None, overwrite_encoded_chunks=False, chunk_store=None, storage_options=None, decode_timedelta=None, use_cftime=None, zarr_version=None, chunked_array_type: str | None=None, from_array_kwargs: dict[str, Any] | None=None, **kwargs):
    """Load and decode a dataset from a Zarr store.

    The `store` object should be a valid store for a Zarr group. `store`
    variables must contain dimension metadata encoded in the
    `_ARRAY_DIMENSIONS` attribute or must have NCZarr format.

    Parameters
    ----------
    store : MutableMapping or str
        A MutableMapping where a Zarr Group has been stored or a path to a
        directory in file system where a Zarr DirectoryStore has been stored.
    synchronizer : object, optional
        Array synchronizer provided to zarr
    group : str, optional
        Group path. (a.k.a. `path` in zarr terminology.)
    chunks : int or dict or tuple or {None, 'auto'}, optional
        Chunk sizes along each dimension, e.g., ``5`` or
        ``{'x': 5, 'y': 5}``. If `chunks='auto'`, dask chunks are created
        based on the variable's zarr chunks. If `chunks=None`, zarr array
        data will lazily convert to numpy arrays upon access. This accepts
        all the chunk specifications as Dask does.
    overwrite_encoded_chunks : bool, optional
        Whether to drop the zarr chunks encoded for each variable when a
        dataset is loaded with specified chunk sizes (default: False)
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    drop_variables : str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    consolidated : bool, optional
        Whether to open the store using zarr's consolidated metadata
        capability. Only works for stores that have already been consolidated.
        By default (`consolidate=None`), attempts to read consolidated metadata,
        falling back to read non-consolidated metadata if that fails.

        When the experimental ``zarr_version=3``, ``consolidated`` must be
        either be ``None`` or ``False``.
    chunk_store : MutableMapping, optional
        A separate Zarr store only for chunk data.
    storage_options : dict, optional
        Any additional parameters for the storage backend (ignored for local
        paths).
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {'days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds'}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.
    use_cftime : bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.
    zarr_version : int or None, optional
        The desired zarr spec version to target (currently 2 or 3). The default
        of None will attempt to determine the zarr version from ``store`` when
        possible, otherwise defaulting to 2.
    chunked_array_type: str, optional
        Which chunked array type to coerce this datasets' arrays to.
        Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEntryPoint` system.
        Experimental API that should not be relied upon.
    from_array_kwargs: dict, optional
        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
        Defaults to {'manager': 'dask'}, meaning additional kwargs will be passed eventually to
        :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    See Also
    --------
    open_dataset
    open_mfdataset

    References
    ----------
    http://zarr.readthedocs.io/
    """
    from xarray.backends.api import open_dataset
    if from_array_kwargs is None:
        from_array_kwargs = {}
    if chunks == 'auto':
        try:
            guess_chunkmanager(chunked_array_type)
            chunks = {}
        except ValueError:
            chunks = None
    if kwargs:
        raise TypeError('open_zarr() got unexpected keyword arguments ' + ','.join(kwargs.keys()))
    backend_kwargs = {'synchronizer': synchronizer, 'consolidated': consolidated, 'overwrite_encoded_chunks': overwrite_encoded_chunks, 'chunk_store': chunk_store, 'storage_options': storage_options, 'stacklevel': 4, 'zarr_version': zarr_version}
    ds = open_dataset(filename_or_obj=store, group=group, decode_cf=decode_cf, mask_and_scale=mask_and_scale, decode_times=decode_times, concat_characters=concat_characters, decode_coords=decode_coords, engine='zarr', chunks=chunks, drop_variables=drop_variables, chunked_array_type=chunked_array_type, from_array_kwargs=from_array_kwargs, backend_kwargs=backend_kwargs, decode_timedelta=decode_timedelta, use_cftime=use_cftime, zarr_version=zarr_version)
    return ds