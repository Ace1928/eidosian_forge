import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_generate_IPv6_with_bad_prefix(self):
    mac = '00:16:3e:33:44:55'
    bad_prefix = 'bb'
    self.assertRaises(ValueError, lambda: netutils.get_ipv6_addr_by_EUI64(bad_prefix, mac))