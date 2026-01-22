from __future__ import print_function, absolute_import, division
import pytest
from .test_core import f, _test_powell
@pytest.mark.skipif(missing_import, reason='pyneqsys.symbolic req. missing')
def test_TransformedSys__from_callback():
    ts = TransformedSys.from_callback(f, (sp.exp, sp.log), 2, 1)
    x, sol = ts.solve([1, 0.1], [3], solver='scipy')
    assert sol['success']
    assert abs(x[0] - 0.8411639) < 2e-07
    assert abs(x[1] - 0.1588361) < 2e-07