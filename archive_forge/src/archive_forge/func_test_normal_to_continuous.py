import numpy as np
from scipy import stats
from tune._utils import (
def test_normal_to_continuous():
    np.random.seed(0)
    values = np.random.normal(0, 1.0, 100000)
    res = normal_to_continuous(values, 10, 15)
    t = stats.kstest(res, 'norm', args=(10, 15))
    assert t.pvalue > 0.4