from netaddr import IPAddress, IPNetwork
def test_ip_v4_to_ipv6_mapped():
    ip = IPAddress('192.0.2.15').ipv6()
    assert ip == IPAddress('::ffff:192.0.2.15')
    assert ip.is_ipv4_mapped()
    assert not ip.is_ipv4_compat()