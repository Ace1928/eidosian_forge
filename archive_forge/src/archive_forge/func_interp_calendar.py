from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import date_range_like, get_date_type
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.coding.times import _should_cftime_be_used, convert_times
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
def interp_calendar(source, target, dim='time'):
    """Interpolates a DataArray or Dataset indexed by a time coordinate to
    another calendar based on decimal year measure.

    Each timestamp in `source` and `target` are first converted to their decimal
    year equivalent then `source` is interpolated on the target coordinate.
    The decimal year of a timestamp is its year plus its sub-year component
    converted to the fraction of its year. For example "2000-03-01 12:00" is
    2000.1653 in a standard calendar or 2000.16301 in a `"noleap"` calendar.

    This method should only be used when the time (HH:MM:SS) information of
    time coordinate is not important.

    Parameters
    ----------
    source: DataArray or Dataset
      The source data to interpolate; must have a time coordinate of a valid
      dtype (:py:class:`numpy.datetime64` or :py:class:`cftime.datetime` objects)
    target: DataArray, DatetimeIndex, or CFTimeIndex
      The target time coordinate of a valid dtype (np.datetime64 or cftime objects)
    dim : str
      The time coordinate name.

    Return
    ------
    DataArray or Dataset
      The source interpolated on the decimal years of target,
    """
    from xarray.core.dataarray import DataArray
    if isinstance(target, (pd.DatetimeIndex, CFTimeIndex)):
        target = DataArray(target, dims=(dim,), name=dim)
    if not _contains_datetime_like_objects(source[dim].variable) or not _contains_datetime_like_objects(target.variable):
        raise ValueError(f"Both 'source.{dim}' and 'target' must contain datetime objects.")
    source_calendar = source[dim].dt.calendar
    target_calendar = target.dt.calendar
    if (source[dim].time.dt.year == 0).any() and target_calendar in _CALENDARS_WITHOUT_YEAR_ZERO:
        raise ValueError(f'Source time coordinate contains dates with year 0, which is not supported by target calendar {target_calendar}.')
    out = source.copy()
    out[dim] = _datetime_to_decimal_year(source[dim], dim=dim, calendar=source_calendar)
    target_idx = _datetime_to_decimal_year(target, dim=dim, calendar=target_calendar)
    out = out.interp(**{dim: target_idx})
    out[dim] = target
    return out