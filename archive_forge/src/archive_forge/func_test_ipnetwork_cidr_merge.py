import types
from netaddr import IPNetwork, cidr_merge
def test_ipnetwork_cidr_merge():
    ip_list = list(IPNetwork('fe80::/120')) + [IPNetwork('192.0.2.0/24'), IPNetwork('192.0.4.0/25'), IPNetwork('192.0.4.128/25')] + list(map(str, IPNetwork('192.0.3.0/24')))
    assert len(ip_list) == 515
    assert cidr_merge(ip_list) == [IPNetwork('192.0.2.0/23'), IPNetwork('192.0.4.0/24'), IPNetwork('fe80::/120')]