import pytest
from netaddr import valid_nmap_range, iter_nmap_range, IPAddress, AddrFormatError
def test_iter_nmap_range_remove_duplicates():
    assert list(iter_nmap_range('10.0.0.42,42-42')) == [IPAddress('10.0.0.42')]