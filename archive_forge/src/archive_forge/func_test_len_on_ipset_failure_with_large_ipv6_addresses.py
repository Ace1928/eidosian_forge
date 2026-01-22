import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_len_on_ipset_failure_with_large_ipv6_addresses():
    s1 = IPSet(IPRange(IPAddress('::0'), IPAddress(sys.maxsize, 6)))
    with pytest.raises(IndexError):
        len(s1)
    s2 = IPSet(IPRange(IPAddress('::0'), IPAddress(sys.maxsize - 1, 6)))
    assert len(s2) == sys.maxsize