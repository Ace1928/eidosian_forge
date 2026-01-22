import pytest
import sysconfig
import numpy as np
from numpy.testing import assert_, assert_raises, IS_WASM
def test_errcall(self):

    def foo(*args):
        print(args)
    olderrcall = np.geterrcall()
    with np.errstate(call=foo):
        assert_(np.geterrcall() is foo, 'call is not foo')
        with np.errstate(call=None):
            assert_(np.geterrcall() is None, 'call is not None')
    assert_(np.geterrcall() is olderrcall, 'call is not olderrcall')