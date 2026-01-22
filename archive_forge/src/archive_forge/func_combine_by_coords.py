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
def combine_by_coords(data_objects: Iterable[Dataset | DataArray]=[], compat: CompatOptions='no_conflicts', data_vars: Literal['all', 'minimal', 'different'] | list[str]='all', coords: str='different', fill_value: object=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='no_conflicts') -> Dataset | DataArray:
    """

    Attempt to auto-magically combine the given datasets (or data arrays)
    into one by using dimension coordinates.

    This function attempts to combine a group of datasets along any number of
    dimensions into a single entity by inspecting coords and metadata and using
    a combination of concat and merge.

    Will attempt to order the datasets such that the values in their dimension
    coordinates are monotonic along all dimensions. If it cannot determine the
    order in which to concatenate the datasets, it will raise a ValueError.
    Non-coordinate dimensions will be ignored, as will any coordinate
    dimensions which do not vary between each dataset.

    Aligns coordinates, but different variables on datasets can cause it
    to fail under some scenarios. In complex cases, you may need to clean up
    your data and use concat/merge explicitly (also see `combine_nested`).

    Works well if, for example, you have N years of data and M data variables,
    and each combination of a distinct time period and set of data variables is
    saved as its own dataset. Also useful for if you have a simulation which is
    parallelized in multiple dimensions, but has global coordinates saved in
    each file specifying the positions of points within the global domain.

    Parameters
    ----------
    data_objects : Iterable of Datasets or DataArrays
        Data objects to combine.

    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential conflicts:

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
        These data variables will be concatenated together:

        - "minimal": Only data variables in which the dimension already
          appears are included.
        - "different": Data variables which are not equal (ignoring
          attributes) across all datasets are also concatenated (as well as
          all for which dimension already appears). Beware: this option may
          load the data payload of data variables into memory if they are not
          already loaded.
        - "all": All data variables will be concatenated.
        - list of str: The listed data variables will be concatenated, in
          addition to the "minimal" data variables.

        If objects are DataArrays, `data_vars` must be "all".
    coords : {"minimal", "different", "all"} or list of str, optional
        As per the "data_vars" kwarg, but for coordinate variables.
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values. If None, raises a ValueError if
        the passed Datasets do not create a complete hypercube.
    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes in objects

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                      "override"} or callable, default: "no_conflicts"
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
    combined : xarray.Dataset or xarray.DataArray
        Will return a Dataset unless all the inputs are unnamed DataArrays, in which case a
        DataArray will be returned.

    See also
    --------
    concat
    merge
    combine_nested

    Examples
    --------

    Combining two datasets using their common dimension coordinates. Notice
    they are concatenated based on the values in their dimension coordinates,
    not on their position in the list passed to `combine_by_coords`.

    >>> x1 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [0, 1], "x": [10, 20, 30]},
    ... )
    >>> x2 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [2, 3], "x": [10, 20, 30]},
    ... )
    >>> x3 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [2, 3], "x": [40, 50, 60]},
    ... )

    >>> x1
    <xarray.Dataset> Size: 136B
    Dimensions:        (y: 2, x: 3)
    Coordinates:
      * y              (y) int64 16B 0 1
      * x              (x) int64 24B 10 20 30
    Data variables:
        temperature    (y, x) float64 48B 10.98 14.3 12.06 10.9 8.473 12.92
        precipitation  (y, x) float64 48B 0.4376 0.8918 0.9637 0.3834 0.7917 0.5289

    >>> x2
    <xarray.Dataset> Size: 136B
    Dimensions:        (y: 2, x: 3)
    Coordinates:
      * y              (y) int64 16B 2 3
      * x              (x) int64 24B 10 20 30
    Data variables:
        temperature    (y, x) float64 48B 11.36 18.51 1.421 1.743 0.4044 16.65
        precipitation  (y, x) float64 48B 0.7782 0.87 0.9786 0.7992 0.4615 0.7805

    >>> x3
    <xarray.Dataset> Size: 136B
    Dimensions:        (y: 2, x: 3)
    Coordinates:
      * y              (y) int64 16B 2 3
      * x              (x) int64 24B 40 50 60
    Data variables:
        temperature    (y, x) float64 48B 2.365 12.8 2.867 18.89 10.44 8.293
        precipitation  (y, x) float64 48B 0.2646 0.7742 0.4562 0.5684 0.01879 0.6176

    >>> xr.combine_by_coords([x2, x1])
    <xarray.Dataset> Size: 248B
    Dimensions:        (y: 4, x: 3)
    Coordinates:
      * y              (y) int64 32B 0 1 2 3
      * x              (x) int64 24B 10 20 30
    Data variables:
        temperature    (y, x) float64 96B 10.98 14.3 12.06 ... 1.743 0.4044 16.65
        precipitation  (y, x) float64 96B 0.4376 0.8918 0.9637 ... 0.4615 0.7805

    >>> xr.combine_by_coords([x3, x1])
    <xarray.Dataset> Size: 464B
    Dimensions:        (y: 4, x: 6)
    Coordinates:
      * y              (y) int64 32B 0 1 2 3
      * x              (x) int64 48B 10 20 30 40 50 60
    Data variables:
        temperature    (y, x) float64 192B 10.98 14.3 12.06 ... 18.89 10.44 8.293
        precipitation  (y, x) float64 192B 0.4376 0.8918 0.9637 ... 0.01879 0.6176

    >>> xr.combine_by_coords([x3, x1], join="override")
    <xarray.Dataset> Size: 256B
    Dimensions:        (y: 2, x: 6)
    Coordinates:
      * y              (y) int64 16B 0 1
      * x              (x) int64 48B 10 20 30 40 50 60
    Data variables:
        temperature    (y, x) float64 96B 10.98 14.3 12.06 ... 18.89 10.44 8.293
        precipitation  (y, x) float64 96B 0.4376 0.8918 0.9637 ... 0.01879 0.6176

    >>> xr.combine_by_coords([x1, x2, x3])
    <xarray.Dataset> Size: 464B
    Dimensions:        (y: 4, x: 6)
    Coordinates:
      * y              (y) int64 32B 0 1 2 3
      * x              (x) int64 48B 10 20 30 40 50 60
    Data variables:
        temperature    (y, x) float64 192B 10.98 14.3 12.06 ... 18.89 10.44 8.293
        precipitation  (y, x) float64 192B 0.4376 0.8918 0.9637 ... 0.01879 0.6176

    You can also combine DataArray objects, but the behaviour will differ depending on
    whether or not the DataArrays are named. If all DataArrays are named then they will
    be promoted to Datasets before combining, and then the resultant Dataset will be
    returned, e.g.

    >>> named_da1 = xr.DataArray(
    ...     name="a", data=[1.0, 2.0], coords={"x": [0, 1]}, dims="x"
    ... )
    >>> named_da1
    <xarray.DataArray 'a' (x: 2)> Size: 16B
    array([1., 2.])
    Coordinates:
      * x        (x) int64 16B 0 1

    >>> named_da2 = xr.DataArray(
    ...     name="a", data=[3.0, 4.0], coords={"x": [2, 3]}, dims="x"
    ... )
    >>> named_da2
    <xarray.DataArray 'a' (x: 2)> Size: 16B
    array([3., 4.])
    Coordinates:
      * x        (x) int64 16B 2 3

    >>> xr.combine_by_coords([named_da1, named_da2])
    <xarray.Dataset> Size: 64B
    Dimensions:  (x: 4)
    Coordinates:
      * x        (x) int64 32B 0 1 2 3
    Data variables:
        a        (x) float64 32B 1.0 2.0 3.0 4.0

    If all the DataArrays are unnamed, a single DataArray will be returned, e.g.

    >>> unnamed_da1 = xr.DataArray(data=[1.0, 2.0], coords={"x": [0, 1]}, dims="x")
    >>> unnamed_da2 = xr.DataArray(data=[3.0, 4.0], coords={"x": [2, 3]}, dims="x")
    >>> xr.combine_by_coords([unnamed_da1, unnamed_da2])
    <xarray.DataArray (x: 4)> Size: 32B
    array([1., 2., 3., 4.])
    Coordinates:
      * x        (x) int64 32B 0 1 2 3

    Finally, if you attempt to combine a mix of unnamed DataArrays with either named
    DataArrays or Datasets, a ValueError will be raised (as this is an ambiguous operation).
    """
    if not data_objects:
        return Dataset()
    objs_are_unnamed_dataarrays = [isinstance(data_object, DataArray) and data_object.name is None for data_object in data_objects]
    if any(objs_are_unnamed_dataarrays):
        if all(objs_are_unnamed_dataarrays):
            temp_datasets = [unnamed_dataarray._to_temp_dataset() for unnamed_dataarray in data_objects]
            combined_temp_dataset = _combine_single_variable_hypercube(temp_datasets, fill_value=fill_value, data_vars=data_vars, coords=coords, compat=compat, join=join, combine_attrs=combine_attrs)
            return DataArray()._from_temp_dataset(combined_temp_dataset)
        else:
            raise ValueError("Can't automatically combine unnamed DataArrays with either named DataArrays or Datasets.")
    else:
        data_objects = [obj.to_dataset() if isinstance(obj, DataArray) else obj for obj in data_objects]
        sorted_datasets = sorted(data_objects, key=vars_as_keys)
        grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)
        concatenated_grouped_by_data_vars = tuple((_combine_single_variable_hypercube(tuple(datasets_with_same_vars), fill_value=fill_value, data_vars=data_vars, coords=coords, compat=compat, join=join, combine_attrs=combine_attrs) for vars, datasets_with_same_vars in grouped_by_vars))
    return merge(concatenated_grouped_by_data_vars, compat=compat, fill_value=fill_value, join=join, combine_attrs=combine_attrs)