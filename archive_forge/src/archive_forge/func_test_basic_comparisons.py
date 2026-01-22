from netaddr import IPAddress, IPNetwork
def test_basic_comparisons():
    assert IPAddress('192.0.2.1') == IPAddress('192.0.2.1')
    assert not IPAddress('192.0.2.1') != IPAddress('192.0.2.1')
    assert IPAddress('192.0.2.2') > IPAddress('192.0.2.1')
    assert IPAddress('192.0.2.1') >= IPAddress('192.0.2.1')
    assert IPAddress('192.0.2.2') >= IPAddress('192.0.2.1')
    assert IPAddress('192.0.2.1') < IPAddress('192.0.2.2')
    assert IPAddress('192.0.2.1') <= IPAddress('192.0.2.1')
    assert IPAddress('192.0.2.1') <= IPAddress('192.0.2.2')