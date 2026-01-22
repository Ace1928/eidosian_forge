import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@mock.patch('uuid.uuid4')
@mock.patch.object(utils.priv_nvme, 'create_hostid')
@mock.patch.object(builtins, 'open')
def test_get_nvme_host_id_io_err_no_uuid(self, mock_open, mock_create, mock_uuid):
    mock_create.return_value = mock.sentinel.uuid_return
    mock_open.side_effect = IOError()
    result = utils.get_nvme_host_id(None)
    mock_open.assert_called_once_with('/etc/nvme/hostid', 'r')
    mock_create.assert_called_once_with(str(mock_uuid.return_value))
    self.assertEqual(mock.sentinel.uuid_return, result)