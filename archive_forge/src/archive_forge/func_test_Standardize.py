from __future__ import print_function
import numpy as np
from patsy.state import Center, Standardize, center
from patsy.util import atleast_2d_column_default
def test_Standardize():
    check_stateful(Standardize, True, [1, -1], [1, -1])
    check_stateful(Standardize, True, [12, 10], [1, -1])
    check_stateful(Standardize, True, [12, 11, 10], [np.sqrt(3.0 / 2), 0, -np.sqrt(3.0 / 2)])
    check_stateful(Standardize, True, [12.0, 11.0, 10.0], [np.sqrt(3.0 / 2), 0, -np.sqrt(3.0 / 2)])
    r20 = list(range(20))
    check_stateful(Standardize, True, [1, -1], [np.sqrt(2) / 2, -np.sqrt(2) / 2], ddof=1)
    check_stateful(Standardize, True, r20, list((np.arange(20) - 9.5) / 5.766281297335398), ddof=0)
    check_stateful(Standardize, True, r20, list((np.arange(20) - 9.5) / 5.916079783099616), ddof=1)
    check_stateful(Standardize, True, r20, list(np.arange(20) - 9.5), rescale=False, ddof=1)
    check_stateful(Standardize, True, r20, list(np.arange(20) / 5.916079783099616), center=False, ddof=1)
    check_stateful(Standardize, True, r20, r20, center=False, rescale=False, ddof=1)