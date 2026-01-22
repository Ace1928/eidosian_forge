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
def compute_ics(y, x, df):
    if x.shape[1]:
        resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    else:
        resid = y
    nobs = resid.shape[0]
    sigma2 = 1.0 / nobs * sumofsq(resid)
    llf = -nobs * (np.log(2 * np.pi * sigma2) + 1) / 2
    res = SimpleNamespace(nobs=nobs, df_model=df + x.shape[1], sigma2=sigma2, llf=llf)
    aic = call_cached_func(ARDLResults.aic, res)
    bic = call_cached_func(ARDLResults.bic, res)
    hqic = call_cached_func(ARDLResults.hqic, res)
    return (aic, bic, hqic)