import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_disjointed_ipsets():
    s1 = IPSet(['192.0.2.0', '192.0.2.1', '192.0.2.2'])
    s2 = IPSet(['192.0.2.2', '192.0.2.3', '192.0.2.4'])
    assert s1 & s2 == IPSet(['192.0.2.2/32'])
    assert not s1.isdisjoint(s2)
    s3 = IPSet(['192.0.2.0', '192.0.2.1'])
    s4 = IPSet(['192.0.2.3', '192.0.2.4'])
    assert s3 & s4 == IPSet([])
    assert s3.isdisjoint(s4)