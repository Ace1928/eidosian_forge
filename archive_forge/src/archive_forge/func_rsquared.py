from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal
from collections.abc import Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
@cache_readonly
def rsquared(self):
    """
        R-squared of the model.

        This is defined here as 1 - `ssr`/`centered_tss` if the constant is
        included in the model and 1 - `ssr`/`uncentered_tss` if the constant is
        omitted.
        """
    if self.k_constant:
        return 1 - self.ssr / self.centered_tss
    else:
        return 1 - self.ssr / self.uncentered_tss