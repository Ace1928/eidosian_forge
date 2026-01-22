import errno
import os
import socket
from unittest import mock
import eventlet
from oslo_service import eventlet_backdoor
from oslo_service.tests import base
@mock.patch.object(os, 'unlink')
@mock.patch.object(eventlet, 'spawn')
@mock.patch.object(eventlet, 'listen')
def test_backdoor_path_already_exists_and_gone(self, listen_mock, spawn_mock, unlink_mock):
    self.config(backdoor_socket='/tmp/my_special_socket')
    sock = mock.Mock()
    listen_mock.side_effect = [socket.error(errno.EADDRINUSE, ''), sock]
    unlink_mock.side_effect = OSError(errno.ENOENT, '')
    path = eventlet_backdoor.initialize_if_enabled(self.conf)
    self.assertEqual('/tmp/my_special_socket', path)
    unlink_mock.assert_called_with('/tmp/my_special_socket')