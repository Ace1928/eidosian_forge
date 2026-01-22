from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_ip_range():
    ip_list = list(iter_iprange('192.0.2.1', '192.0.2.14'))
    assert len(ip_list) == 14
    assert ip_list == [IPAddress('192.0.2.1'), IPAddress('192.0.2.2'), IPAddress('192.0.2.3'), IPAddress('192.0.2.4'), IPAddress('192.0.2.5'), IPAddress('192.0.2.6'), IPAddress('192.0.2.7'), IPAddress('192.0.2.8'), IPAddress('192.0.2.9'), IPAddress('192.0.2.10'), IPAddress('192.0.2.11'), IPAddress('192.0.2.12'), IPAddress('192.0.2.13'), IPAddress('192.0.2.14')]
    assert cidr_merge(ip_list) == [IPNetwork('192.0.2.1/32'), IPNetwork('192.0.2.2/31'), IPNetwork('192.0.2.4/30'), IPNetwork('192.0.2.8/30'), IPNetwork('192.0.2.12/31'), IPNetwork('192.0.2.14/32')]