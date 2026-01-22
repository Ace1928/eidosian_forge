import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_invalid_icmp_type(self):
    invalid_inputs = [-1, '-1', 256, '256', None, 'None', 'five']
    for input_value in invalid_inputs:
        self.assertFalse(netutils.is_valid_icmp_type(input_value))