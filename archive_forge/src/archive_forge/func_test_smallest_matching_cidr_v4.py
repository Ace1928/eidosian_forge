import random
from netaddr import (
def test_smallest_matching_cidr_v4():
    assert smallest_matching_cidr('192.0.2.0', ['10.0.0.1', '192.0.2.0', '224.0.0.1']) == IPNetwork('192.0.2.0/32')
    assert smallest_matching_cidr('192.0.2.32', ['0.0.0.0/0', '10.0.0.0/8', '192.0.0.0/8', '192.0.1.0/24', '192.0.2.0/24', '192.0.3.0/24']) == IPNetwork('192.0.2.0/24')
    assert smallest_matching_cidr('192.0.2.0', ['10.0.0.1', '224.0.0.1']) is None