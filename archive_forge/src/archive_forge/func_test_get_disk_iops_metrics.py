from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(metricsutils.MetricsUtils, '_get_metrics_values')
@mock.patch.object(metricsutils.MetricsUtils, '_get_vm_resources')
def test_get_disk_iops_metrics(self, mock_get_vm_resources, mock_get_metrics_values):
    self.utils._metrics_defs_obj = {self.utils._DISK_IOPS_METRICS: mock.sentinel.metrics}
    mock_disk = mock.MagicMock(HostResource=[mock.sentinel.host_resource], InstanceID=mock.sentinel.instance_id)
    mock_get_vm_resources.return_value = [mock_disk]
    mock_get_metrics_values.return_value = [mock.sentinel.iops]
    disk_metrics = list(self.utils.get_disk_iops_count(mock.sentinel.vm_name))
    self.assertEqual(1, len(disk_metrics))
    self.assertEqual(mock.sentinel.iops, disk_metrics[0]['iops_count'])
    self.assertEqual(mock.sentinel.instance_id, disk_metrics[0]['instance_id'])
    mock_get_vm_resources.assert_called_once_with(mock.sentinel.vm_name, self.utils._STORAGE_ALLOC_SETTING_DATA_CLASS)
    mock_get_metrics_values.assert_called_once_with(mock_disk, [mock.sentinel.metrics])