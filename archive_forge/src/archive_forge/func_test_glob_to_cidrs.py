from netaddr import (
def test_glob_to_cidrs():
    assert glob_to_cidrs('10.0.0.1') == [IPNetwork('10.0.0.1/32')]
    assert glob_to_cidrs('192.0.2.*') == [IPNetwork('192.0.2.0/24')]
    assert glob_to_cidrs('172.16-31.*.*') == [IPNetwork('172.16.0.0/12')]
    assert glob_to_cidrs('*.*.*.*') == [IPNetwork('0.0.0.0/0')]