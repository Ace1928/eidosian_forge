import pytest
from netaddr import valid_nmap_range, iter_nmap_range, IPAddress, AddrFormatError
def test_valid_nmap_range_with_invalid_target_specs():
    assert not valid_nmap_range('192.0.2.0/255.255.255.0')
    assert not valid_nmap_range(1)
    assert not valid_nmap_range('1')
    assert not valid_nmap_range([])
    assert not valid_nmap_range({})
    assert not valid_nmap_range('fe80::/64')
    assert not valid_nmap_range('255.255.255.256')
    assert not valid_nmap_range('0-255.0-255.0-255.0-256')
    assert not valid_nmap_range('0-255.0-255.0-255.-1-0')
    assert not valid_nmap_range('0-255.0-255.0-255.256-0')
    assert not valid_nmap_range('0-255.0-255.0-255.255-0')
    assert not valid_nmap_range('a.b.c.d-e')
    assert not valid_nmap_range('255.255.255.a-b')