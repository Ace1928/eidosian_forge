import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_integer_constructor_v6():
    assert IPAddress(8589934591) == IPAddress('::1:ffff:ffff')
    assert IPAddress(4294967295, 6) == IPAddress('::255.255.255.255')
    assert IPAddress(8589934591) == IPAddress('::1:ffff:ffff')
    assert IPAddress(2 ** 128 - 1) == IPAddress('ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff')