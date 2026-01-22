import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_get_my_ipv4(self):
    sock_attrs = {'return_value.getsockname.return_value': ['1.2.3.4', '']}
    with mock.patch('socket.socket', **sock_attrs):
        addr = netutils.get_my_ipv4()
    self.assertEqual(addr, '1.2.3.4')