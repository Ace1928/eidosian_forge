from netaddr import IPNetwork
def test_is_loopback():
    assert IPNetwork('127.0.0.0/8').is_loopback()
    assert IPNetwork('::1/128').is_loopback()