import os
import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tsa.vector_ar.vecm import coint_johansen
Test Johansen's Cointegration test against jplv, Spatial Econometrics Toolbox

Created on Thu Aug 30 21:51:08 2012
Author: Josef Perktold

