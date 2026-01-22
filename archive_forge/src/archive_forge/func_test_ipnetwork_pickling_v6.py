import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_ipnetwork_pickling_v6():
    cidr = IPNetwork('::ffff:192.0.2.0/120')
    assert cidr == IPNetwork('::ffff:192.0.2.0/120')
    assert cidr.value == 281473902969344
    assert cidr.prefixlen == 120
    buf = pickle.dumps(cidr)
    cidr2 = pickle.loads(buf)
    assert cidr2 == cidr
    assert cidr2.value == 281473902969344
    assert cidr2.prefixlen == 120
    assert cidr2.version == 6