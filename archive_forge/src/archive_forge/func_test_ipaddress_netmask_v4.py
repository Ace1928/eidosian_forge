import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_netmask_v4():
    assert IPAddress('0.0.0.0').netmask_bits() == 0
    assert IPAddress('128.0.0.0').netmask_bits() == 1
    assert IPAddress('255.0.0.0').netmask_bits() == 8
    assert IPAddress('255.255.0.0').netmask_bits() == 16
    assert IPAddress('255.255.255.0').netmask_bits() == 24
    assert IPAddress('255.255.255.254').netmask_bits() == 31
    assert IPAddress('255.255.255.255').netmask_bits() == 32
    assert IPAddress('1.1.1.1').netmask_bits() == 32