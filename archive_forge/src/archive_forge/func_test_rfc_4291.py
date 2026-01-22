from netaddr import iprange_to_cidrs, IPNetwork, cidr_merge, all_matching_cidrs
def test_rfc_4291():
    assert str(IPNetwork('2001:0DB8:0000:CD30:0000:0000:0000:0000/60')) == '2001:db8:0:cd30::/60'
    assert str(IPNetwork('2001:0DB8::CD30:0:0:0:0/60')) == '2001:db8:0:cd30::/60'
    assert str(IPNetwork('2001:0DB8:0:CD30::/60')) == '2001:db8:0:cd30::/60'