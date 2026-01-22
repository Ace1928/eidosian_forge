import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_empty():
    assert IPSet() == IPSet([])
    empty_set = IPSet([])
    assert IPSet([]) == empty_set
    assert len(empty_set) == 0