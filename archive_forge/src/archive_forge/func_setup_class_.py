from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
@classmethod
def setup_class_(cls):
    cls.mc = MultiComparison(cls.endog, cls.groups)
    cls.res = cls.mc.tukeyhsd(alpha=cls.alpha)