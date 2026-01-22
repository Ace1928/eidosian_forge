import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
@mock.patch('netifaces.gateways')
@mock.patch('netifaces.ifaddresses')
def test_get_my_ip_address_with_default_route(self, ifaddr, gateways):
    ifaddr.return_value = {netifaces.AF_INET: [{'addr': '172.18.204.1'}], netifaces.AF_INET6: [{'addr': '2001:db8::2'}]}
    self.assertEqual('172.18.204.1', netutils._get_my_ipv4_address())
    self.assertEqual('2001:db8::2', netutils._get_my_ipv6_address())