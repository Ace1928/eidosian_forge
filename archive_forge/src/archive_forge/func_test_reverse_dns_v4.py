from netaddr import IPAddress
def test_reverse_dns_v4():
    assert IPAddress('172.24.0.13').reverse_dns == '13.0.24.172.in-addr.arpa.'