import random
from netaddr import (
def test_cidr_matching_v4():
    networks = [str(c) for c in IPNetwork('192.0.2.128/27').supernet(22)]
    assert networks == ['192.0.0.0/22', '192.0.2.0/23', '192.0.2.0/24', '192.0.2.128/25', '192.0.2.128/26']
    assert all_matching_cidrs('192.0.2.0', networks) == [IPNetwork('192.0.0.0/22'), IPNetwork('192.0.2.0/23'), IPNetwork('192.0.2.0/24')]
    assert smallest_matching_cidr('192.0.2.0', networks) == IPNetwork('192.0.2.0/24')
    assert largest_matching_cidr('192.0.2.0', networks) == IPNetwork('192.0.0.0/22')