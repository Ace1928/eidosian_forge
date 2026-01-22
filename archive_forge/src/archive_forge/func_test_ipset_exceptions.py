import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_exceptions():
    s1 = IPSet(['10.0.0.1'])
    with pytest.raises(TypeError):
        hash(s1)
    with pytest.raises(TypeError):
        s1.update(42)