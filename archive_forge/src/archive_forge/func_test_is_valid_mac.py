import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_is_valid_mac(self):
    self.assertTrue(netutils.is_valid_mac('52:54:00:cf:2d:31'))
    self.assertFalse(netutils.is_valid_mac('127.0.0.1'))
    self.assertFalse(netutils.is_valid_mac('not:a:mac:address'))
    self.assertFalse(netutils.is_valid_mac('52-54-00-cf-2d-31'))
    self.assertFalse(netutils.is_valid_mac('aa bb cc dd ee ff'))
    self.assertTrue(netutils.is_valid_mac('AA:BB:CC:DD:EE:FF'))
    self.assertFalse(netutils.is_valid_mac('AA BB CC DD EE FF'))
    self.assertFalse(netutils.is_valid_mac('AA-BB-CC-DD-EE-FF'))