import platform
from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
@mock.patch.object(livemigrationutils.LiveMigrationUtils, '_get_vm')
@mock.patch.object(livemigrationutils.LiveMigrationUtils, '_get_ip_address_list')
@mock.patch.object(livemigrationutils.LiveMigrationUtils, '_update_planned_vm_disk_resources')
@mock.patch.object(livemigrationutils.LiveMigrationUtils, '_create_planned_vm')
@mock.patch.object(livemigrationutils.LiveMigrationUtils, 'destroy_existing_planned_vm')
@mock.patch.object(livemigrationutils.LiveMigrationUtils, '_get_disk_data')
def test_create_planned_vm(self, mock_get_disk_data, mock_destroy_existing_planned_vm, mock_create_planned_vm, mock_update_planned_vm_disk_resources, mock_get_ip_address_list, mock_get_vm):
    dest_host = platform.node()
    mock_vm = mock.MagicMock()
    mock_get_vm.return_value = mock_vm
    mock_conn_v2 = mock.MagicMock()
    self.liveutils._get_wmi_obj.return_value = mock_conn_v2
    mock_get_disk_data.return_value = mock.sentinel.disk_data
    mock_get_ip_address_list.return_value = mock.sentinel.ip_address_list
    mock_vsmsvc = self._conn.Msvm_VirtualSystemManagementService()[0]
    mock_vsmsvc.ModifyResourceSettings.return_value = (mock.sentinel.res_setting, mock.sentinel.job_path, self._FAKE_RET_VAL)
    self.liveutils.create_planned_vm(mock.sentinel.vm_name, mock.sentinel.host, mock.sentinel.disk_path_mapping)
    mock_destroy_existing_planned_vm.assert_called_once_with(mock.sentinel.vm_name)
    mock_get_ip_address_list.assert_called_once_with(self._conn, dest_host)
    mock_get_disk_data.assert_called_once_with(mock.sentinel.vm_name, vmutils.VMUtils.return_value, mock.sentinel.disk_path_mapping)
    mock_create_planned_vm.assert_called_once_with(self._conn, mock_conn_v2, mock_vm, mock.sentinel.ip_address_list, dest_host)
    mock_update_planned_vm_disk_resources.assert_called_once_with(self._conn, mock_create_planned_vm.return_value, mock.sentinel.vm_name, mock.sentinel.disk_data)