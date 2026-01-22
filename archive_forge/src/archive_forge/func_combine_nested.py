from __future__ import annotations
import itertools
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.concat import concat
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.merge import merge
from xarray.core.utils import iterate_nested
def combine_nested(datasets: DATASET_HYPERCUBE, concat_dim: str | DataArray | None | Sequence[str | DataArray | pd.Index | None], compat: str='no_conflicts', data_vars: str='all', coords: str='different', fill_value: object=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='drop') -> Dataset:
    """
    Explicitly combine an N-dimensional grid of datasets into one by using a
    succession of concat and merge operations along each dimension of the grid.

    Does not sort the supplied datasets under any circumstances, so the
    datasets must be passed in the order you wish them to be concatenated. It
    does align coordinates, but different variables on datasets can cause it to
    fail under some scenarios. In complex cases, you may need to clean up your
    data and use concat/merge explicitly.

    To concatenate along multiple dimensions the datasets must be passed as a
    nested list-of-lists, with a depth equal to the length of ``concat_dims``.
    ``combine_nested`` will concatenate along the top-level list first.

    Useful for combining datasets from a set of nested directories, or for
    collecting the output of a simulation parallelized along multiple
    dimensions.

    Parameters
    ----------
    datasets : list or nested list of Dataset
        Dataset objects to combine.
        If concatenation or merging along more than one dimension is desired,
        then datasets must be supplied in a nested list-of-lists.
    concat_dim : str, or list of str, DataArray, Index or None
        Dimensions along which to concatenate variables, as used by
        :py:func:`xarray.concat`.
        Set ``concat_dim=[..., None, ...]`` explicitly to disable concatenation
        and merge instead along a particular dimension.
        The position of ``None`` in the list specifies the dimension of the
        nested-list input along which to merge.
        Must be the same length as the depth of the list passed to
        ``datasets``.
    compat : {"identical", "equals", "broadcast_equals",               "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential merge conflicts:

        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset
    data_vars : {"minimal", "different", "all" or list of str}, optional
        Details are in the documentation of concat
    coords : {"minimal", "different", "all" or list of str}, optional
        Details are in the documentation of concat
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.
    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                      "override"} or callable, default: "drop"
        A callable or a string indicating how to combine attrs of the objects being
        merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "drop_conflicts": attrs from all objects are combined, any that have
          the same name but different values are dropped.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.

        If a callable, it must expect a sequence of ``attrs`` dicts and a context object
        as its only parameters.

    Returns
    -------
    combined : xarray.Dataset

    Examples
    --------

    A common task is collecting data from a parallelized simulation in which
    each process wrote out to a separate file. A domain which was decomposed
    into 4 parts, 2 each along both the x and y axes, requires organising the
    datasets into a doubly-nested list, e.g:

    >>> x1y1 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )
    >>> x1y1
    <xarray.Dataset> Size: 64B
    Dimensions:        (x: 2, y: 2)
    Dimensions without coordinates: x, y
    Data variables:
        temperature    (x, y) float64 32B 1.764 0.4002 0.9787 2.241
        precipitation  (x, y) float64 32B 1.868 -0.9773 0.9501 -0.1514
    >>> x1y2 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )
    >>> x2y1 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )
    >>> x2y2 = xr.Dataset(
    ...     {
    ...         "temperature": (("x", "y"), np.random.randn(2, 2)),
    ...         "precipitation": (("x", "y"), np.random.randn(2, 2)),
    ...     }
    ... )


    >>> ds_grid = [[x1y1, x1y2], [x2y1, x2y2]]
    >>> combined = xr.combine_nested(ds_grid, concat_dim=["x", "y"])
    >>> combined
    <xarray.Dataset> Size: 256B
    Dimensions:        (x: 4, y: 4)
    Dimensions without coordinates: x, y
    Data variables:
        temperature    (x, y) float64 128B 1.764 0.4002 -0.1032 ... 0.04576 -0.1872
        precipitation  (x, y) float64 128B 1.868 -0.9773 0.761 ... 0.1549 0.3782

    ``combine_nested`` can also be used to explicitly merge datasets with
    different variables. For example if we have 4 datasets, which are divided
    along two times, and contain two different variables, we can pass ``None``
    to ``concat_dim`` to specify the dimension of the nested list over which
    we wish to use ``merge`` instead of ``concat``:

    >>> t1temp = xr.Dataset({"temperature": ("t", np.random.randn(5))})
    >>> t1temp
    <xarray.Dataset> Size: 40B
    Dimensions:      (t: 5)
    Dimensions without coordinates: t
    Data variables:
        temperature  (t) float64 40B -0.8878 -1.981 -0.3479 0.1563 1.23

    >>> t1precip = xr.Dataset({"precipitation": ("t", np.random.randn(5))})
    >>> t1precip
    <xarray.Dataset> Size: 40B
    Dimensions:        (t: 5)
    Dimensions without coordinates: t
    Data variables:
        precipitation  (t) float64 40B 1.202 -0.3873 -0.3023 -1.049 -1.42

    >>> t2temp = xr.Dataset({"temperature": ("t", np.random.randn(5))})
    >>> t2precip = xr.Dataset({"precipitation": ("t", np.random.randn(5))})


    >>> ds_grid = [[t1temp, t1precip], [t2temp, t2precip]]
    >>> combined = xr.combine_nested(ds_grid, concat_dim=["t", None])
    >>> combined
    <xarray.Dataset> Size: 160B
    Dimensions:        (t: 10)
    Dimensions without coordinates: t
    Data variables:
        temperature    (t) float64 80B -0.8878 -1.981 -0.3479 ... -0.4381 -1.253
        precipitation  (t) float64 80B 1.202 -0.3873 -0.3023 ... -0.8955 0.3869

    See also
    --------
    concat
    merge
    """
    mixed_datasets_and_arrays = any((isinstance(obj, Dataset) for obj in iterate_nested(datasets))) and any((isinstance(obj, DataArray) and obj.name is None for obj in iterate_nested(datasets)))
    if mixed_datasets_and_arrays:
        raise ValueError("Can't combine datasets with unnamed arrays.")
    if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
        concat_dim = [concat_dim]
    return _nested_combine(datasets, concat_dims=concat_dim, compat=compat, data_vars=data_vars, coords=coords, ids=False, fill_value=fill_value, join=join, combine_attrs=combine_attrs)