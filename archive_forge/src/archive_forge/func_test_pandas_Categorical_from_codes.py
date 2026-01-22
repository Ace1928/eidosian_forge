import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_pandas_Categorical_from_codes():
    if not have_pandas_categorical:
        return
    c = pandas_Categorical_from_codes([1, 1, 0, -1], ['a', 'b'])
    assert np.all(np.asarray(c)[:-1] == ['b', 'b', 'a'])
    assert np.isnan(np.asarray(c)[-1])