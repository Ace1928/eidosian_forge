import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_unions_intersections_differences():
    adj_cidrs = list(IPNetwork('192.0.2.0/24').subnet(28))
    even_cidrs = adj_cidrs[::2]
    evens = IPSet(even_cidrs)
    assert evens == IPSet(['192.0.2.0/28', '192.0.2.32/28', '192.0.2.64/28', '192.0.2.96/28', '192.0.2.128/28', '192.0.2.160/28', '192.0.2.192/28', '192.0.2.224/28'])
    assert IPSet(['192.0.2.0/24']) & evens == IPSet(['192.0.2.0/28', '192.0.2.32/28', '192.0.2.64/28', '192.0.2.96/28', '192.0.2.128/28', '192.0.2.160/28', '192.0.2.192/28', '192.0.2.224/28'])
    odds = IPSet(['192.0.2.0/24']) ^ evens
    assert odds == IPSet(['192.0.2.16/28', '192.0.2.48/28', '192.0.2.80/28', '192.0.2.112/28', '192.0.2.144/28', '192.0.2.176/28', '192.0.2.208/28', '192.0.2.240/28'])
    assert evens | odds == IPSet(['192.0.2.0/24'])
    assert evens & odds == IPSet([])
    assert evens ^ odds == IPSet(['192.0.2.0/24'])