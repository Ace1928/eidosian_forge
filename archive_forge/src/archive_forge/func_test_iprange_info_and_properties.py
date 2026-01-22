from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_info_and_properties():
    iprange = IPRange('192.0.2.1', '192.0.2.254')
    assert literal_eval(str(iprange.info)) == {'IPv4': [{'date': '1993-05', 'designation': 'Administered by ARIN', 'prefix': '192/8', 'status': 'Legacy', 'whois': 'whois.arin.net'}]}
    assert iprange.is_reserved()
    assert iprange.version == 4