import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_ip_network_equality_v6():
    assert IPNetwork('fe80::/10') == IPNetwork('fe80::/10')
    assert IPNetwork('fe80::/10') is not IPNetwork('fe80::/10')
    assert not IPNetwork('fe80::/10') != IPNetwork('fe80::/10')
    assert not IPNetwork('fe80::/10') is IPNetwork('fe80::/10')