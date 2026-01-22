import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from statsmodels import datasets
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel

Tests for CFA simulation smoothing in a TVP-VAR model

See "results/cfa_tvpvar_test.m" for Matlab file that generates the results.

Based on "TVPVAR.m", found at http://joshuachan.org/code/code_TVPVAR.html.
See [1]_ for details on the TVP-VAR model and the CFA method.

References
----------
.. [1] Chan, Joshua CC, and Ivan Jeliazkov.
       "Efficient simulation and integrated likelihood estimation in
       state space models."
       International Journal of Mathematical Modelling and Numerical
       Optimisation 1, no. 1-2 (2009): 101-120.

Author: Chad Fulton
License: BSD-3
