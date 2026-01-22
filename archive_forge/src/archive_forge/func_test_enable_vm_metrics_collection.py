from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(metricsutils.MetricsUtils, '_enable_metrics')
@mock.patch.object(metricsutils.MetricsUtils, '_get_vm_resources')
@mock.patch.object(metricsutils.MetricsUtils, '_get_vm')
def test_enable_vm_metrics_collection(self, mock_get_vm, mock_get_vm_resources, mock_enable_metrics):
    mock_vm = mock_get_vm.return_value
    mock_disk = mock.MagicMock()
    mock_dvd = mock.MagicMock(ResourceSubType=self.utils._DVD_DISK_RES_SUB_TYPE)
    mock_get_vm_resources.return_value = [mock_disk, mock_dvd]
    self.utils.enable_vm_metrics_collection(mock.sentinel.vm_name)
    metrics_names = [self.utils._CPU_METRICS, self.utils._MEMORY_METRICS]
    mock_enable_metrics.assert_has_calls([mock.call(mock_disk), mock.call(mock_vm, metrics_names)])