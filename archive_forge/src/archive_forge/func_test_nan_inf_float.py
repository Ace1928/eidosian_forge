import sys
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, IS_MUSL
from numpy.core.tests._locales import CommaDecimalPointLocale
from io import StringIO
@pytest.mark.parametrize('tp', [np.float32, np.double, np.longdouble])
def test_nan_inf_float(tp):
    """ Check formatting of nan & inf.

        This is only for the str function, and only for simple types.
        The precision of np.float32 and np.longdouble aren't the same as the
        python float precision.

    """
    for x in [np.inf, -np.inf, np.nan]:
        assert_equal(str(tp(x)), _REF[x], err_msg='Failed str formatting for type %s' % tp)