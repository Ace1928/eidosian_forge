import errno
import os
import socket
from unittest import mock
import eventlet
from oslo_service import eventlet_backdoor
from oslo_service.tests import base
@mock.patch.object(eventlet, 'spawn')
@mock.patch.object(eventlet, 'listen')
def test_backdoor_port_range_one_inuse(self, listen_mock, spawn_mock):
    self.config(backdoor_port='8800:8900')
    sock = mock.Mock()
    sock.getsockname.return_value = ('127.0.0.1', 8801)
    listen_mock.side_effect = [socket.error(errno.EADDRINUSE, ''), sock]
    port = eventlet_backdoor.initialize_if_enabled(self.conf)
    self.assertEqual(8801, port)