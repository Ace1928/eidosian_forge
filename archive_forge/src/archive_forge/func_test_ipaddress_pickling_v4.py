import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_pickling_v4():
    ip = IPAddress(3221225985)
    assert ip == IPAddress('192.0.2.1')
    buf = pickle.dumps(ip)
    ip2 = pickle.loads(buf)
    assert ip2 == ip
    assert id(ip2) != id(ip)
    assert ip2.value == 3221225985
    assert ip2.version == 4