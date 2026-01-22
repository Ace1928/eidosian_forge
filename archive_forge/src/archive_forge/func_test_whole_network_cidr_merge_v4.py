import random
from netaddr import (
def test_whole_network_cidr_merge_v4():
    assert cidr_merge(['0.0.0.0/0', '0.0.0.0']) == [IPNetwork('0.0.0.0/0')]
    assert cidr_merge(['0.0.0.0/0', '255.255.255.255']) == [IPNetwork('0.0.0.0/0')]
    assert cidr_merge(['0.0.0.0/0', '192.0.2.0/24', '10.0.0.0/8']) == [IPNetwork('0.0.0.0/0')]