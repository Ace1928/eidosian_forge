import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_binary_operations_v4():
    assert IPAddress('192.0.2.15') & IPAddress('255.255.255.0') == IPAddress('192.0.2.0')
    assert IPAddress('255.255.0.0') | IPAddress('0.0.255.255') == IPAddress('255.255.255.255')
    assert IPAddress('255.255.0.0') ^ IPAddress('255.0.0.0') == IPAddress('0.255.0.0')
    assert IPAddress('1.2.3.4').packed == '\x01\x02\x03\x04'.encode('ascii')