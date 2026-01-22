import pickle
import types
import random
import pytest
from netaddr import (
def test_iterhosts_v4():
    assert list(IPNetwork('192.0.2.0/29').iter_hosts()) == [IPAddress('192.0.2.1'), IPAddress('192.0.2.2'), IPAddress('192.0.2.3'), IPAddress('192.0.2.4'), IPAddress('192.0.2.5'), IPAddress('192.0.2.6')]
    assert list(IPNetwork('192.168.0.0/31')) == [IPAddress('192.168.0.0'), IPAddress('192.168.0.1')]
    assert list(IPNetwork('192.168.0.0/31').iter_hosts()) == [IPAddress('192.168.0.0'), IPAddress('192.168.0.1')]
    assert list(IPNetwork('192.168.0.0/32').iter_hosts()) == [IPAddress('192.168.0.0')]