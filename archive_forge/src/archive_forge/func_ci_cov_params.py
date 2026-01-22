from __future__ import annotations
from statsmodels.compat.pandas import Appender, Substitution, call_cached_func
from collections import defaultdict
import datetime as dt
from itertools import combinations, product
import textwrap
from types import SimpleNamespace
from typing import (
from collections.abc import Hashable, Mapping, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary, summary_params
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.ardl import pss_critical_values
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.tsatools import lagmat
from_formula_doc = Docstring(ARDL.from_formula.__doc__)
from_formula_doc.replace_block("Summary", "Construct an UECM from a formula")
from_formula_doc.remove_parameters("lags")
from_formula_doc.remove_parameters("order")
from_formula_doc.insert_parameters("data", lags_param)
from_formula_doc.insert_parameters("lags", order_param)
def ci_cov_params(self) -> Float64Array | pd.DataFrame:
    """Covariance of normalized of cointegrating relationship"""
    ndet = self.model._blocks['deterministic'].shape[1]
    nlvl = self.model._blocks['levels'].shape[1]
    loc = list(range(ndet + nlvl))
    cov = self.cov_params()
    cov_a = np.asarray(cov)
    ci_cov = cov_a[np.ix_(loc, loc)]
    m = ci_cov.shape[0]
    params = np.asarray(self.params)[:ndet + nlvl]
    base = params[ndet]
    d = np.zeros((m, m))
    for i in range(m):
        if i == ndet:
            continue
        d[i, i] = 1 / base
        d[i, ndet] = -params[i] / base ** 2
    ci_cov = d @ ci_cov @ d.T
    return self._ci_wrap(ci_cov)