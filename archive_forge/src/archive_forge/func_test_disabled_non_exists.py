import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
@mock.patch('os.path.exists', return_value=False)
@mock.patch('builtins.open', side_effect=AssertionError('should not read'))
def test_disabled_non_exists(self, mock_open, exists):
    enabled = netutils.is_ipv6_enabled()
    self.assertFalse(enabled)