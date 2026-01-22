from __future__ import annotations
import os
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from functools import partial
from io import BytesIO
from numbers import Number
from typing import (
import numpy as np
from xarray import backends, conventions
from xarray.backends import plugins
from xarray.backends.common import (
from xarray.backends.locks import _get_scheduler
from xarray.backends.zarr import open_zarr
from xarray.core import indexing
from xarray.core.combine import (
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, _get_chunk, _maybe_chunk
from xarray.core.indexes import Index
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import is_remote_uri
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import guess_chunkmanager
def save_mfdataset(datasets, paths, mode='w', format=None, groups=None, engine=None, compute=True, **kwargs):
    """Write multiple datasets to disk as netCDF files simultaneously.

    This function is intended for use with datasets consisting of dask.array
    objects, in which case it can write the multiple datasets to disk
    simultaneously using a shared thread pool.

    When not using dask, it is no different than calling ``to_netcdf``
    repeatedly.

    Parameters
    ----------
    datasets : list of Dataset
        List of datasets to save.
    paths : list of str or list of path-like objects
        List of paths to which to save each corresponding dataset.
    mode : {"w", "a"}, optional
        Write ("w") or append ("a") mode. If mode="w", any existing file at
        these locations will be overwritten.
    format : {"NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT",               "NETCDF3_CLASSIC"}, optional
    **kwargs : additional arguments are passed along to ``to_netcdf``

        File format for the resulting netCDF file:

        * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API
          features.
        * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only
          netCDF 3 compatible API features.
        * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,
          which fully supports 2+ GB files, but is only compatible with
          clients linked against netCDF version 3.6.0 or later.
        * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not
          handle 2+ GB files very well.

        All formats are supported by the netCDF4-python library.
        scipy.io.netcdf only supports the last two formats.

        The default format is NETCDF4 if you are saving a file to disk and
        have the netCDF4-python library available. Otherwise, xarray falls
        back to using scipy to write netCDF files and defaults to the
        NETCDF3_64BIT format (scipy does not support netCDF4).
    groups : list of str, optional
        Paths to the netCDF4 group in each corresponding file to which to save
        datasets (only works for format="NETCDF4"). The groups will be created
        if necessary.
    engine : {"netcdf4", "scipy", "h5netcdf"}, optional
        Engine to use when writing netCDF files. If not provided, the
        default engine is chosen based on available dependencies, with a
        preference for "netcdf4" if writing to a file on disk.
        See `Dataset.to_netcdf` for additional information.
    compute : bool
        If true compute immediately, otherwise return a
        ``dask.delayed.Delayed`` object that can be computed later.

    Examples
    --------

    Save a dataset into one netCDF per year of data:

    >>> ds = xr.Dataset(
    ...     {"a": ("time", np.linspace(0, 1, 48))},
    ...     coords={"time": pd.date_range("2010-01-01", freq="ME", periods=48)},
    ... )
    >>> ds
    <xarray.Dataset> Size: 768B
    Dimensions:  (time: 48)
    Coordinates:
      * time     (time) datetime64[ns] 384B 2010-01-31 2010-02-28 ... 2013-12-31
    Data variables:
        a        (time) float64 384B 0.0 0.02128 0.04255 ... 0.9574 0.9787 1.0
    >>> years, datasets = zip(*ds.groupby("time.year"))
    >>> paths = [f"{y}.nc" for y in years]
    >>> xr.save_mfdataset(datasets, paths)
    """
    if mode == 'w' and len(set(paths)) < len(paths):
        raise ValueError("cannot use mode='w' when writing multiple datasets to the same path")
    for obj in datasets:
        if not isinstance(obj, Dataset):
            raise TypeError(f'save_mfdataset only supports writing Dataset objects, received type {type(obj)}')
    if groups is None:
        groups = [None] * len(datasets)
    if len({len(datasets), len(paths), len(groups)}) > 1:
        raise ValueError('must supply lists of the same length for the datasets, paths and groups arguments to save_mfdataset')
    writers, stores = zip(*[to_netcdf(ds, path, mode, format, group, engine, compute=compute, multifile=True, **kwargs) for ds, path, group in zip(datasets, paths, groups)])
    try:
        writes = [w.sync(compute=compute) for w in writers]
    finally:
        if compute:
            for store in stores:
                store.close()
    if not compute:
        import dask
        return dask.delayed([dask.delayed(_finalize_store)(w, s) for w, s in zip(writes, stores)])