from netaddr import IPNetwork
def test_is_unicast():
    assert IPNetwork('192.0.2.0/24').is_unicast()
    assert IPNetwork('fe80::1/48').is_unicast()