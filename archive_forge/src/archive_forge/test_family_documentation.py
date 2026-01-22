import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.tools.sm_exceptions import (
import statsmodels.genmod.families as F
from statsmodels.genmod.families.family import Tweedie
import statsmodels.genmod.families.links as L
Test that Tweedie loglike is normalized to 1.