import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_valid_icmp_code(self):
    valid_inputs = [1, '1', 0, '0', 255, '255', None]
    for input_value in valid_inputs:
        self.assertTrue(netutils.is_valid_icmp_code(input_value))