import pytest
from netaddr.ip import IPNetwork
from netaddr.contrib.subnet_splitter import SubnetSplitter
def test_ip_splitter_remove_same_input_range():
    s = SubnetSplitter('172.24.0.0/16')
    assert s.available_subnets() == [IPNetwork('172.24.0.0/16')]
    assert s.extract_subnet(16, count=1) == [IPNetwork('172.24.0.0/16')]
    assert s.available_subnets() == []