import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_inet_pton_constructor_v4_accepts_valid_input():
    assert IPAddress('10.0.0.1', flags=INET_PTON) == IPAddress('10.0.0.1')