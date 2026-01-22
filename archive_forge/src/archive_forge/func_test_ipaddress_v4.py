import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_v4():
    ip = IPAddress('192.0.2.1')
    assert ip.version == 4
    assert repr(ip) == "IPAddress('192.0.2.1')"
    assert str(ip) == '192.0.2.1'
    assert ip.format() == '192.0.2.1'
    assert int(ip) == 3221225985
    assert hex(ip) == '0xc0000201'
    assert bytes(ip) == b'\xc0\x00\x02\x01'
    assert ip.bin == '0b11000000000000000000001000000001'
    assert ip.bits() == '11000000.00000000.00000010.00000001'
    assert ip.words == (192, 0, 2, 1)