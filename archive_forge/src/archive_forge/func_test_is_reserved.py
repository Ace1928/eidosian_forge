from netaddr import IPNetwork
def test_is_reserved():
    assert IPNetwork('240.0.0.0/24').is_reserved()
    assert IPNetwork('0::/48').is_reserved()