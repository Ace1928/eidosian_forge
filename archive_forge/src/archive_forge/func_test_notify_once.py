import os
import socket
from unittest import mock
from oslotest import base as test_base
from oslo_service import systemd
def test_notify_once(self):
    os.environ['NOTIFY_SOCKET'] = '@fake_socket'
    self._test__sd_notify(unset_env=True)
    self.assertRaises(KeyError, os.environ.__getitem__, 'NOTIFY_SOCKET')