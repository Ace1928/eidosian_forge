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
def news(self, comparison, impact_date=None, impacted_variable=None, start=None, end=None, periods=None, exog=None, comparison_type=None, revisions_details_start=False, state_index=None, return_raw=False, tolerance=1e-10, endog_quarterly=None, original_scale=True, **kwargs):
    """
        Compute impacts from updated data (news and revisions).

        Parameters
        ----------
        comparison : array_like or MLEResults
            An updated dataset with updated and/or revised data from which the
            news can be computed, or an updated or previous results object
            to use in computing the news.
        impact_date : int, str, or datetime, optional
            A single specific period of impacts from news and revisions to
            compute. Can also be a date string to parse or a datetime type.
            This argument cannot be used in combination with `start`, `end`, or
            `periods`. Default is the first out-of-sample observation.
        impacted_variable : str, list, array, or slice, optional
            Observation variable label or slice of labels specifying that only
            specific impacted variables should be shown in the News output. The
            impacted variable(s) describe the variables that were *affected* by
            the news. If you do not know the labels for the variables, check
            the `endog_names` attribute of the model instance.
        start : int, str, or datetime, optional
            The first period of impacts from news and revisions to compute.
            Can also be a date string to parse or a datetime type. Default is
            the first out-of-sample observation.
        end : int, str, or datetime, optional
            The last period of impacts from news and revisions to compute.
            Can also be a date string to parse or a datetime type. Default is
            the first out-of-sample observation.
        periods : int, optional
            The number of periods of impacts from news and revisions to
            compute.
        exog : array_like, optional
            Array of exogenous regressors for the out-of-sample period, if
            applicable.
        comparison_type : {None, 'previous', 'updated'}
            This denotes whether the `comparison` argument represents a
            *previous* results object or dataset or an *updated* results object
            or dataset. If not specified, then an attempt is made to determine
            the comparison type.
        state_index : array_like or "common", optional
            An optional index specifying a subset of states to use when
            constructing the impacts of revisions and news. For example, if
            `state_index=[0, 1]` is passed, then only the impacts to the
            observed variables arising from the impacts to the first two
            states will be returned. If the string "common" is passed and the
            model includes idiosyncratic AR(1) components, news will only be
            computed based on the common states. Default is to use all states.
        return_raw : bool, optional
            Whether or not to return only the specific output or a full
            results object. Default is to return a full results object.
        tolerance : float, optional
            The numerical threshold for determining zero impact. Default is
            that any impact less than 1e-10 is assumed to be zero.
        endog_quarterly : array_like, optional
            New observations of quarterly variables, if `comparison` was
            provided as an updated monthly dataset. If this argument is
            provided, it must be a Pandas Series or DataFrame with a
            DatetimeIndex or PeriodIndex at the quarterly frequency.

        References
        ----------
        .. [1] Bańbura, Marta, and Michele Modugno.
               "Maximum likelihood estimation of factor models on datasets with
               arbitrary pattern of missing data."
               Journal of Applied Econometrics 29, no. 1 (2014): 133-160.
        .. [2] Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin.
               "Nowcasting."
               The Oxford Handbook of Economic Forecasting. July 8, 2011.
        .. [3] Bańbura, Marta, Domenico Giannone, Michele Modugno, and Lucrezia
               Reichlin.
               "Now-casting and the real-time data flow."
               In Handbook of economic forecasting, vol. 2, pp. 195-237.
               Elsevier, 2013.
        """
    if state_index == 'common':
        state_index = np.arange(self.model.k_states - self.model.k_endog)
    news_results = super().news(comparison, impact_date=impact_date, impacted_variable=impacted_variable, start=start, end=end, periods=periods, exog=exog, comparison_type=comparison_type, revisions_details_start=revisions_details_start, state_index=state_index, return_raw=return_raw, tolerance=tolerance, endog_quarterly=endog_quarterly, **kwargs)
    if not return_raw and self.model.standardize and original_scale:
        endog_mean = self.model._endog_mean
        endog_std = self.model._endog_std
        news_results.total_impacts = news_results.total_impacts * endog_std
        news_results.update_impacts = news_results.update_impacts * endog_std
        if news_results.revision_impacts is not None:
            news_results.revision_impacts = news_results.revision_impacts * endog_std
        if news_results.revision_detailed_impacts is not None:
            news_results.revision_detailed_impacts = news_results.revision_detailed_impacts * endog_std
        if news_results.revision_grouped_impacts is not None:
            news_results.revision_grouped_impacts = news_results.revision_grouped_impacts * endog_std
        for name in ['prev_impacted_forecasts', 'news', 'revisions', 'update_realized', 'update_forecasts', 'revised', 'revised_prev', 'post_impacted_forecasts', 'revisions_all', 'revised_all', 'revised_prev_all']:
            dta = getattr(news_results, name)
            orig_name = None
            if hasattr(dta, 'name'):
                orig_name = dta.name
            dta = dta.multiply(endog_std, level=1)
            if name not in ['news', 'revisions']:
                dta = dta.add(endog_mean, level=1)
            if orig_name is not None:
                dta.name = orig_name
            setattr(news_results, name, dta)
        news_results.weights = news_results.weights.divide(endog_std, axis=0, level=1).multiply(endog_std, axis=1, level=1)
        news_results.revision_weights = news_results.revision_weights.divide(endog_std, axis=0, level=1).multiply(endog_std, axis=1, level=1)
    return news_results