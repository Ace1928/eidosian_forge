import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.stats.proportion as smprop
from statsmodels.stats.proportion import (
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.stats.tests.results.results_proportion import res_binom, res_binom_methods

    diff:
    Wald 0.029 0.32 0.29
    Agresti–Caffo 0.012 0.32 0.31
    Newcombe hybrid score 0.019 0.34 0.32
    Miettinen–Nurminen asymptotic score 0.028 0.34 0.31
    Santner–Snell exact unconditional -0.069 0.41 0.48
    Chan–Zhang exact unconditional 0.019 0.36 0.34
    Agresti–Min exact unconditional 0.024 0.35 0.33

    ratio:
    Katz log 0.91 54 4.08
    Adjusted log 0.92 27 3.38
    Inverse sinh 1.17 42 3.58
    Koopman asymptotic score 1.21 43 3.57
    Chan–Zhang 1.22 181 5.00
    Agresti–Min 1.15 89 4.35

    odds-ratio
    Woolf logit 0.99 74 4.31
    Gart adjusted logit 0.98 38 3.65
    Independence-smoothed logit 0.99 60 4.11
    Cornfield exact conditional 0.97 397 6.01
    Cornfield mid-p 1.19 200 5.12
    Baptista–Pike exact conditional 1.00 195 5.28
    Baptista–Pike mid-p 1.33 99 4.31
    Agresti–Min exact unconditional 1.19 72 4.10
    