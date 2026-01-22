from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.missing import (
def quantile_with_mask(values: np.ndarray, mask: npt.NDArray[np.bool_], fill_value, qs: npt.NDArray[np.float64], interpolation: str) -> np.ndarray:
    """
    Compute the quantiles of the given values for each quantile in `qs`.

    Parameters
    ----------
    values : np.ndarray
        For ExtensionArray, this is _values_for_factorize()[0]
    mask : np.ndarray[bool]
        mask = isna(values)
        For ExtensionArray, this is computed before calling _value_for_factorize
    fill_value : Scalar
        The value to interpret fill NA entries with
        For ExtensionArray, this is _values_for_factorize()[1]
    qs : np.ndarray[float64]
    interpolation : str
        Type of interpolation

    Returns
    -------
    np.ndarray

    Notes
    -----
    Assumes values is already 2D.  For ExtensionArray this means np.atleast_2d
    has been called on _values_for_factorize()[0]

    Quantile is computed along axis=1.
    """
    assert values.shape == mask.shape
    if values.ndim == 1:
        values = np.atleast_2d(values)
        mask = np.atleast_2d(mask)
        res_values = quantile_with_mask(values, mask, fill_value, qs, interpolation)
        return res_values[0]
    assert values.ndim == 2
    is_empty = values.shape[1] == 0
    if is_empty:
        flat = np.array([fill_value] * len(qs))
        result = np.repeat(flat, len(values)).reshape(len(values), len(qs))
    else:
        result = _nanpercentile(values, qs * 100.0, na_value=fill_value, mask=mask, interpolation=interpolation)
        result = np.array(result, copy=False)
        result = result.T
    return result