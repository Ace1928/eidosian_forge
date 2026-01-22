import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_is_valid_ip(self):
    self.assertTrue(netutils.is_valid_ip('127.0.0.1'))
    self.assertTrue(netutils.is_valid_ip('2001:db8::ff00:42:8329'))
    self.assertTrue(netutils.is_valid_ip('fe80::1%eth0'))
    self.assertFalse(netutils.is_valid_ip('256.0.0.0'))
    self.assertFalse(netutils.is_valid_ip('::1.2.3.'))
    self.assertFalse(netutils.is_valid_ip(''))
    self.assertFalse(netutils.is_valid_ip(None))