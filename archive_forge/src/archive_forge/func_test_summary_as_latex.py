from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_summary_as_latex():
    import re
    dta = longley.load_pandas()
    x = dta.exog
    x['constant'] = 1
    y = dta.endog
    res = OLS(y, x).fit()
    with pytest.warns(UserWarning):
        table = res.summary().as_latex()
    table = re.sub('(?<=\n\\\\textbf\\{Date:\\}             &).+?&', ' Sun, 07 Apr 2013 &', table)
    table = re.sub('(?<=\n\\\\textbf\\{Time:\\}             &).+?&', '     13:46:07     &', table)
    expected = '\\begin{center}\n\\begin{tabular}{lclc}\n\\toprule\n\\textbf{Dep. Variable:}    &      TOTEMP      & \\textbf{  R-squared:         } &     0.995   \\\\\n\\textbf{Model:}            &       OLS        & \\textbf{  Adj. R-squared:    } &     0.992   \\\\\n\\textbf{Method:}           &  Least Squares   & \\textbf{  F-statistic:       } &     330.3   \\\\\n\\textbf{Date:}             & Sun, 07 Apr 2013 & \\textbf{  Prob (F-statistic):} &  4.98e-10   \\\\\n\\textbf{Time:}             &     13:46:07     & \\textbf{  Log-Likelihood:    } &   -109.62   \\\\\n\\textbf{No. Observations:} &          16      & \\textbf{  AIC:               } &     233.2   \\\\\n\\textbf{Df Residuals:}     &           9      & \\textbf{  BIC:               } &     238.6   \\\\\n\\textbf{Df Model:}         &           6      & \\textbf{                     } &             \\\\\n\\textbf{Covariance Type:}  &    nonrobust     & \\textbf{                     } &             \\\\\n\\bottomrule\n\\end{tabular}\n\\begin{tabular}{lcccccc}\n                  & \\textbf{coef} & \\textbf{std err} & \\textbf{t} & \\textbf{P$> |$t$|$} & \\textbf{[0.025} & \\textbf{0.975]}  \\\\\n\\midrule\n\\textbf{GNPDEFL}  &      15.0619  &       84.915     &     0.177  &         0.863        &     -177.029    &      207.153     \\\\\n\\textbf{GNP}      &      -0.0358  &        0.033     &    -1.070  &         0.313        &       -0.112    &        0.040     \\\\\n\\textbf{UNEMP}    &      -2.0202  &        0.488     &    -4.136  &         0.003        &       -3.125    &       -0.915     \\\\\n\\textbf{ARMED}    &      -1.0332  &        0.214     &    -4.822  &         0.001        &       -1.518    &       -0.549     \\\\\n\\textbf{POP}      &      -0.0511  &        0.226     &    -0.226  &         0.826        &       -0.563    &        0.460     \\\\\n\\textbf{YEAR}     &    1829.1515  &      455.478     &     4.016  &         0.003        &      798.788    &     2859.515     \\\\\n\\textbf{constant} &   -3.482e+06  &      8.9e+05     &    -3.911  &         0.004        &     -5.5e+06    &    -1.47e+06     \\\\\n\\bottomrule\n\\end{tabular}\n\\begin{tabular}{lclc}\n\\textbf{Omnibus:}       &  0.749 & \\textbf{  Durbin-Watson:     } &    2.559  \\\\\n\\textbf{Prob(Omnibus):} &  0.688 & \\textbf{  Jarque-Bera (JB):  } &    0.684  \\\\\n\\textbf{Skew:}          &  0.420 & \\textbf{  Prob(JB):          } &    0.710  \\\\\n\\textbf{Kurtosis:}      &  2.434 & \\textbf{  Cond. No.          } & 4.86e+09  \\\\\n\\bottomrule\n\\end{tabular}\n%\\caption{OLS Regression Results}\n\\end{center}\n\nNotes: \\newline\n [1] Standard Errors assume that the covariance matrix of the errors is correctly specified. \\newline\n [2] The condition number is large, 4.86e+09. This might indicate that there are \\newline\n strong multicollinearity or other numerical problems.'
    assert_equal(table, expected)