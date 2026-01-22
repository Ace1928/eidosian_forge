from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.windows import base as base_win_conn
from os_brick.tests.windows import fake_win_conn
from os_brick.tests.windows import test_base
def test_get_scsi_wwn(self):
    mock_get_uid_and_type = self._diskutils.get_disk_uid_and_uid_type
    mock_get_uid_and_type.return_value = (mock.sentinel.disk_uid, mock.sentinel.uid_type)
    scsi_wwn = self._connector._get_scsi_wwn(mock.sentinel.dev_num)
    expected_wwn = '%s%s' % (mock.sentinel.uid_type, mock.sentinel.disk_uid)
    self.assertEqual(expected_wwn, scsi_wwn)
    mock_get_uid_and_type.assert_called_once_with(mock.sentinel.dev_num)