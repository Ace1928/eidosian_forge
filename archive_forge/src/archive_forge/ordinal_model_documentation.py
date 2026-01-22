import warnings
from statsmodels.compat.pandas import Appender
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import stats
from statsmodels.base.model import (
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
probability residual

        Probability-scale residual is ``P(Y < y) − P(Y > y)`` where `Y` is the
        observed choice and ``y`` is a random variable corresponding to the
        predicted distribution.

        References
        ----------
        Shepherd BE, Li C, Liu Q (2016) Probability-scale residuals for
        continuous, discrete, and censored data.
        The Canadian Journal of Statistics. 44:463–476.

        Li C and Shepherd BE (2012) A new residual for ordinal outcomes.
        Biometrika. 99: 473–480

        