from __future__ import print_function, absolute_import, division
import pytest
from .test_core import f, _test_powell
@pytest.mark.skipif(missing_import, reason='pyneqsys.symbolic req. missing')
def test_symbolicsys__from_callback__no_params():

    def _nf(x):
        return f(x, [3])
    ss = SymbolicSys.from_callback(_nf, 2)
    x, sol = ss.solve([0.7, 0.3], solver='scipy')
    assert sol['success']
    assert abs(x[0] - 0.8411639) < 2e-07
    assert abs(x[1] - 0.1588361) < 2e-07