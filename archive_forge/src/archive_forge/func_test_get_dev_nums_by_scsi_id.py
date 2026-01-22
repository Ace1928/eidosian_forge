from unittest import mock
import ddt
from os_win import exceptions as os_win_exc
from os_brick import exception
from os_brick.initiator.windows import fibre_channel as fc
from os_brick.tests.windows import test_base
@mock.patch.object(fc.WindowsFCConnector, '_get_fc_hba_wwn_for_port')
def test_get_dev_nums_by_scsi_id(self, mock_get_fc_hba_wwn):
    fake_identifier = dict(id=mock.sentinel.id, type=mock.sentinel.type)
    mock_get_fc_hba_wwn.return_value = mock.sentinel.local_wwnn
    self._fc_utils.get_scsi_device_identifiers.return_value = [fake_identifier]
    self._diskutils.get_disk_numbers_by_unique_id.return_value = mock.sentinel.dev_nums
    dev_nums = self._connector._get_dev_nums_by_scsi_id(mock.sentinel.local_wwpn, mock.sentinel.remote_wwpn, mock.sentinel.fcp_lun)
    self.assertEqual(mock.sentinel.dev_nums, dev_nums)
    mock_get_fc_hba_wwn.assert_called_once_with(mock.sentinel.local_wwpn)
    self._fc_utils.get_scsi_device_identifiers.assert_called_once_with(mock.sentinel.local_wwnn, mock.sentinel.local_wwpn, mock.sentinel.remote_wwpn, mock.sentinel.fcp_lun)
    self._diskutils.get_disk_numbers_by_unique_id.assert_called_once_with(unique_id=mock.sentinel.id, unique_id_format=mock.sentinel.type)