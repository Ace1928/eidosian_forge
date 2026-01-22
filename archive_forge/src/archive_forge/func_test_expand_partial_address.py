import pytest
from netaddr import cidr_abbrev_to_verbose
from netaddr.strategy.ipv4 import expand_partial_address
def test_expand_partial_address():
    assert expand_partial_address('10') == '10.0.0.0'
    assert expand_partial_address('10.1') == '10.1.0.0'
    assert expand_partial_address('192.168.1') == '192.168.1.0'