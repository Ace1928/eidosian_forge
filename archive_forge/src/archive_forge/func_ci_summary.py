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
def ci_summary(self, alpha: float=0.05) -> Summary:

    def _ci(alpha=alpha):
        return np.asarray(self.ci_conf_int(alpha))
    smry = Summary()
    ndet = self.model._blocks['deterministic'].shape[1]
    nlvl = self.model._blocks['levels'].shape[1]
    exog_names = list(self.model.exog_names)[:ndet + nlvl]
    model = SimpleNamespace(endog_names=self.model.endog_names, exog_names=exog_names)
    data = SimpleNamespace(params=self.ci_params, bse=self.ci_bse, tvalues=self.ci_tvalues, pvalues=self.ci_pvalues, conf_int=_ci, model=model)
    tab = summary_params(data)
    tab.title = 'Cointegrating Vector'
    smry.tables.append(tab)
    return smry