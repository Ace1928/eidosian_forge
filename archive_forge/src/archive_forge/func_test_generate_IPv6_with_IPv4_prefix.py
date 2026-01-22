import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_generate_IPv6_with_IPv4_prefix(self):
    ipv4_prefix = '10.0.8'
    mac = '00:16:3e:33:44:55'
    self.assertRaises(ValueError, lambda: netutils.get_ipv6_addr_by_EUI64(ipv4_prefix, mac))