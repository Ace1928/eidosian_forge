import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_is_valid_ipv6(self):
    self.assertTrue(netutils.is_valid_ipv6('::1'))
    self.assertTrue(netutils.is_valid_ipv6('fe80::1%eth0'))
    self.assertFalse(netutils.is_valid_ip('fe%80::1%eth0'))
    self.assertFalse(netutils.is_valid_ipv6('1fff::a88:85a3::172.31.128.1'))
    self.assertFalse(netutils.is_valid_ipv6(''))