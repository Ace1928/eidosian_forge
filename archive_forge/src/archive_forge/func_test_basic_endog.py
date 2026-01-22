from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def test_basic_endog():
    assert_raises(ValueError, MLEModel, endog=1, k_states=1)
    assert_raises(ValueError, MLEModel, endog='a', k_states=1)
    assert_raises(ValueError, MLEModel, endog=True, k_states=1)
    mod = MLEModel([1], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])
    mod = MLEModel([1.0], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])
    mod = MLEModel([True], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])
    mod = MLEModel(['a'], **kwargs)
    assert_raises(ValueError, mod.filter, [])
    endog = [1.0, 2.0]
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = [[1.0], [2.0]]
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = (1.0, 2.0)
    mod = check_endog(endog, **kwargs)
    mod.filter([])