from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_pickling_v6():
    iprange = IPRange('::ffff:192.0.2.1', '::ffff:192.0.2.254')
    assert iprange == IPRange('::ffff:192.0.2.1', '::ffff:192.0.2.254')
    assert iprange.first == 281473902969345
    assert iprange.last == 281473902969598
    assert iprange.version == 6
    buf = pickle.dumps(iprange)
    iprange2 = pickle.loads(buf)
    assert iprange2 == iprange
    assert iprange2.first == 281473902969345
    assert iprange2.last == 281473902969598
    assert iprange2.version == 6