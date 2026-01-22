import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_comparison_with_int_is_invalid():
    s1 = IPSet(['10.0.0.1'])
    assert not s1 == 42
    s1 != 42