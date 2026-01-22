import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_is_valid_ipv6_cidr(self):
    self.assertTrue(netutils.is_valid_ipv6_cidr('2600::/64'))
    self.assertTrue(netutils.is_valid_ipv6_cidr('abcd:ef01:2345:6789:abcd:ef01:192.168.254.254/48'))
    self.assertTrue(netutils.is_valid_ipv6_cidr('0000:0000:0000:0000:0000:0000:0000:0001/32'))
    self.assertTrue(netutils.is_valid_ipv6_cidr('0000:0000:0000:0000:0000:0000:0000:0001'))
    self.assertFalse(netutils.is_valid_ipv6_cidr('foo'))
    self.assertFalse(netutils.is_valid_ipv6_cidr('127.0.0.1'))