from netaddr import IPAddress, IPNetwork
def test_ip_v4_to_ipv6_compatible():
    assert IPAddress('192.0.2.15').ipv6(ipv4_compatible=True) == IPAddress('::192.0.2.15')
    assert IPAddress('192.0.2.15').ipv6(ipv4_compatible=True).is_ipv4_compat()
    assert IPAddress('192.0.2.15').ipv6(True) == IPAddress('::192.0.2.15')
    ip = IPNetwork('192.0.2.1/23')
    assert ip.ipv4() == IPNetwork('192.0.2.1/23')
    assert ip.ipv6() == IPNetwork('::ffff:192.0.2.1/119')
    assert ip.ipv6(ipv4_compatible=True) == IPNetwork('::192.0.2.1/119')