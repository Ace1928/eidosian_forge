import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_valid_port(self):
    valid_inputs = [0, '0', 1, '1', 2, '3', '5', 8, 13, 21, '80', '3246', '65535']
    for input_str in valid_inputs:
        self.assertTrue(netutils.is_valid_port(input_str))