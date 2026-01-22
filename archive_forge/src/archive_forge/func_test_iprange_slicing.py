from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_slicing():
    iprange = IPRange('192.0.2.1', '192.0.2.254')
    assert list(iprange[0:3]) == [IPAddress('192.0.2.1'), IPAddress('192.0.2.2'), IPAddress('192.0.2.3')]
    assert list(iprange[0:10:2]) == [IPAddress('192.0.2.1'), IPAddress('192.0.2.3'), IPAddress('192.0.2.5'), IPAddress('192.0.2.7'), IPAddress('192.0.2.9')]
    assert list(iprange[0:1024:512]) == [IPAddress('192.0.2.1')]