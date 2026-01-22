import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
Test that MICEData does not throw a SettingWithCopyWarning when imputing (https://github.com/statsmodels/statsmodels/issues/5430)