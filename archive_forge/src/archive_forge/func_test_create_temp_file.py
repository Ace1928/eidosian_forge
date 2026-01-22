from unittest import mock
import ddt
from oslo_utils import units
from oslo_vmware.objects import datastore
from oslo_vmware import vim_util
from os_brick import exception
from os_brick.initiator.connectors import vmware
from os_brick.tests.initiator import test_connector
@mock.patch('oslo_utils.fileutils.ensure_tree')
@mock.patch('tempfile.mkstemp')
@mock.patch('os.close')
def test_create_temp_file(self, close, mkstemp, ensure_tree):
    fd = mock.sentinel.fd
    tmp = mock.sentinel.tmp
    mkstemp.return_value = (fd, tmp)
    prefix = '.vmdk'
    suffix = 'test'
    ret = self._connector._create_temp_file(prefix=prefix, suffix=suffix)
    self.assertEqual(tmp, ret)
    ensure_tree.assert_called_once_with(self._connector._tmp_dir)
    mkstemp.assert_called_once_with(dir=self._connector._tmp_dir, prefix=prefix, suffix=suffix)
    close.assert_called_once_with(fd)