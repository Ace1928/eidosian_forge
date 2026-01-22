from unittest import mock
import ddt
from os_brick.initiator.connectors import base_rbd
from os_brick.tests import base
def test_get_rbd_args_with_conf(self):
    res = self._conn._get_rbd_args(self.connection_properties, mock.sentinel.conf_path)
    expected = ['--id', self.user, '--mon_host', self.hosts[0] + ':' + self.ports[0], '--conf', mock.sentinel.conf_path]
    self.assertEqual(expected, res)