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
def test_get_my_ip_address_without_default_route(self, ifaddr, gateways):
    ifaddr.return_value = {}
    self.assertEqual('127.0.0.1', netutils._get_my_ipv4_address())
    self.assertEqual('::1', netutils._get_my_ipv6_address())