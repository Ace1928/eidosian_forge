import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_comparison():
    s1 = IPSet(['fc00::/2'])
    s2 = IPSet(['fc00::/3'])
    assert s1 > s2
    assert not s1 < s2
    assert s1 != s2