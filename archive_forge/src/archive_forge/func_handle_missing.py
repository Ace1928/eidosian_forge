from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
@classmethod
def handle_missing(cls, endog, exog, missing, **kwargs):
    """
        This returns a dictionary with keys endog, exog and the keys of
        kwargs. It preserves Nones.
        """
    none_array_names = []
    missing_idx = kwargs.pop('missing_idx', None)
    if missing_idx is not None:
        combined = ()
        combined_names = []
        if exog is None:
            none_array_names += ['exog']
    elif exog is not None:
        combined = (endog, exog)
        combined_names = ['endog', 'exog']
    else:
        combined = (endog,)
        combined_names = ['endog']
        none_array_names += ['exog']
    combined_2d = ()
    combined_2d_names = []
    if len(kwargs):
        for key, value_array in kwargs.items():
            if value_array is None or np.ndim(value_array) == 0:
                none_array_names += [key]
                continue
            if value_array.ndim == 1:
                combined += (np.asarray(value_array),)
                combined_names += [key]
            elif value_array.squeeze().ndim == 1:
                combined += (np.asarray(value_array),)
                combined_names += [key]
            elif value_array.ndim == 2:
                combined_2d += (np.asarray(value_array),)
                combined_2d_names += [key]
            else:
                raise ValueError('Arrays with more than 2 dimensions are not yet handled')
    if missing_idx is not None:
        nan_mask = missing_idx
        updated_row_mask = None
        if combined:
            combined_nans = _nan_rows(*combined)
            if combined_nans.shape[0] != nan_mask.shape[0]:
                raise ValueError('Shape mismatch between endog/exog and extra arrays given to model.')
            updated_row_mask = combined_nans[~nan_mask]
            nan_mask |= combined_nans
        if combined_2d:
            combined_2d_nans = _nan_rows(combined_2d)
            if combined_2d_nans.shape[0] != nan_mask.shape[0]:
                raise ValueError('Shape mismatch between endog/exog and extra 2d arrays given to model.')
            if updated_row_mask is not None:
                updated_row_mask |= combined_2d_nans[~nan_mask]
            else:
                updated_row_mask = combined_2d_nans[~nan_mask]
            nan_mask |= combined_2d_nans
    else:
        nan_mask = _nan_rows(*combined)
        if combined_2d:
            nan_mask = _nan_rows(*(nan_mask[:, None],) + combined_2d)
    if not np.any(nan_mask):
        combined = dict(zip(combined_names, combined))
        if combined_2d:
            combined.update(dict(zip(combined_2d_names, combined_2d)))
        if none_array_names:
            combined.update({k: kwargs.get(k, None) for k in none_array_names})
        if missing_idx is not None:
            combined.update({'endog': endog})
            if exog is not None:
                combined.update({'exog': exog})
        return (combined, [])
    elif missing == 'raise':
        raise MissingDataError('NaNs were encountered in the data')
    elif missing == 'drop':
        nan_mask = ~nan_mask
        drop_nans = lambda x: cls._drop_nans(x, nan_mask)
        drop_nans_2d = lambda x: cls._drop_nans_2d(x, nan_mask)
        combined = dict(zip(combined_names, lmap(drop_nans, combined)))
        if missing_idx is not None:
            if updated_row_mask is not None:
                updated_row_mask = ~updated_row_mask
                endog = cls._drop_nans(endog, updated_row_mask)
                if exog is not None:
                    exog = cls._drop_nans(exog, updated_row_mask)
            combined.update({'endog': endog})
            if exog is not None:
                combined.update({'exog': exog})
        if combined_2d:
            combined.update(dict(zip(combined_2d_names, lmap(drop_nans_2d, combined_2d))))
        if none_array_names:
            combined.update({k: kwargs.get(k, None) for k in none_array_names})
        return (combined, np.where(~nan_mask)[0].tolist())
    else:
        raise ValueError('missing option %s not understood' % missing)