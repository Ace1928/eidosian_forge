from statsmodels.compat.pandas import MONTH_END, QUARTER_END
from collections import OrderedDict
from warnings import warn
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.base.data import PandasData
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params
@classmethod
def construct_endog(cls, endog_monthly, endog_quarterly):
    """
        Construct a combined dataset from separate monthly and quarterly data.

        Parameters
        ----------
        endog_monthly : array_like
            Monthly dataset. If a quarterly dataset is given, then this must
            be a Pandas object with a PeriodIndex or DatetimeIndex at a monthly
            frequency.
        endog_quarterly : array_like or None
            Quarterly dataset. If not None, then this must be a Pandas object
            with a PeriodIndex or DatetimeIndex at a quarterly frequency.

        Returns
        -------
        endog : array_like
            If both endog_monthly and endog_quarterly were given, this is a
            Pandas DataFrame with a PeriodIndex at the monthly frequency, with
            all of the columns from `endog_monthly` ordered first and the
            columns from `endog_quarterly` ordered afterwards. Otherwise it is
            simply the input `endog_monthly` dataset.
        k_endog_monthly : int
            The number of monthly variables (which are ordered first) in the
            returned `endog` dataset.
        """
    if endog_quarterly is not None:
        base_msg = 'If given both monthly and quarterly data then the monthly dataset must be a Pandas object with a date index at a monthly frequency.'
        if not isinstance(endog_monthly, (pd.Series, pd.DataFrame)):
            raise ValueError('Given monthly dataset is not a Pandas object. ' + base_msg)
        elif endog_monthly.index.inferred_type not in ('datetime64', 'period'):
            raise ValueError('Given monthly dataset has an index with non-date values. ' + base_msg)
        elif not getattr(endog_monthly.index, 'freqstr', 'N')[0] == 'M':
            freqstr = getattr(endog_monthly.index, 'freqstr', 'None')
            raise ValueError(f'Index of given monthly dataset has a non-monthly frequency (to check this, examine the `freqstr` attribute of the index of the dataset - it should start with M if it is monthly). Got {freqstr}. ' + base_msg)
        base_msg = 'If a quarterly dataset is given, then it must be a Pandas object with a date index at a quarterly frequency.'
        if not isinstance(endog_quarterly, (pd.Series, pd.DataFrame)):
            raise ValueError('Given quarterly dataset is not a Pandas object. ' + base_msg)
        elif endog_quarterly.index.inferred_type not in ('datetime64', 'period'):
            raise ValueError('Given quarterly dataset has an index with non-date values. ' + base_msg)
        elif not getattr(endog_quarterly.index, 'freqstr', 'N')[0] == 'Q':
            freqstr = getattr(endog_quarterly.index, 'freqstr', 'None')
            raise ValueError(f'Index of given quarterly dataset has a non-quarterly frequency (to check this, examine the `freqstr` attribute of the index of the dataset - it should start with Q if it is quarterly). Got {freqstr}. ' + base_msg)
        if hasattr(endog_monthly.index, 'to_period'):
            endog_monthly = endog_monthly.to_period('M')
        if hasattr(endog_quarterly.index, 'to_period'):
            endog_quarterly = endog_quarterly.to_period('Q')
        quarterly_resamp = endog_quarterly.copy()
        quarterly_resamp.index = quarterly_resamp.index.to_timestamp()
        quarterly_resamp = quarterly_resamp.resample(QUARTER_END).first()
        quarterly_resamp = quarterly_resamp.resample(MONTH_END).first()
        quarterly_resamp.index = quarterly_resamp.index.to_period()
        endog = pd.concat([endog_monthly, quarterly_resamp], axis=1)
        column_counts = endog.columns.value_counts()
        if column_counts.max() > 1:
            columns = endog.columns.values.astype(object)
            for name in column_counts.index:
                count = column_counts.loc[name]
                if count == 1:
                    continue
                mask = columns == name
                columns[mask] = [f'{name}{i + 1}' for i in range(count)]
            endog.columns = columns
    else:
        endog = endog_monthly.copy()
    shape = endog_monthly.shape
    k_endog_monthly = shape[1] if len(shape) == 2 else 1
    return (endog, k_endog_monthly)