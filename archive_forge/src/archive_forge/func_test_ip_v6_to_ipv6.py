from netaddr import IPAddress, IPNetwork
def test_ip_v6_to_ipv6():
    assert IPNetwork('::ffff:192.0.2.1/119').ipv6() == IPNetwork('::ffff:192.0.2.1/119')