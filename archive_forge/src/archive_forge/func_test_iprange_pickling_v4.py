from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_pickling_v4():
    iprange = IPRange('192.0.2.1', '192.0.2.254')
    assert iprange == IPRange('192.0.2.1', '192.0.2.254')
    assert iprange.first == 3221225985
    assert iprange.last == 3221226238
    assert iprange.version == 4
    buf = pickle.dumps(iprange)
    iprange2 = pickle.loads(buf)
    assert iprange2 == iprange
    assert id(iprange2) != id(iprange)
    assert iprange2.first == 3221225985
    assert iprange2.last == 3221226238
    assert iprange2.version == 4