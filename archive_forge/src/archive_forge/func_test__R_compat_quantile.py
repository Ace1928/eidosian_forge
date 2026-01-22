import numpy as np
from patsy.util import have_pandas, no_pickling, assert_no_pickling
from patsy.state import stateful_transform
def test__R_compat_quantile():

    def t(x, prob, expected):
        assert np.allclose(_R_compat_quantile(x, prob), expected)
    t([10, 20], 0.5, 15)
    t([10, 20], 0.3, 13)
    t([10, 20], [0.3, 0.7], [13, 17])
    t(list(range(10)), [0.3, 0.7], [2.7, 6.3])