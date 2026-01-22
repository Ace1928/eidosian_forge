import pickle
import types
import random
import pytest
from netaddr import (
def test_ip_network_equality_v4():
    assert IPNetwork('192.0.2.0/24') == IPNetwork('192.0.2.0/24')
    assert IPNetwork('192.0.2.0/24') is not IPNetwork('192.0.2.0/24')
    assert not IPNetwork('192.0.2.0/24') != IPNetwork('192.0.2.0/24')
    assert not IPNetwork('192.0.2.0/24') is IPNetwork('192.0.2.0/24')