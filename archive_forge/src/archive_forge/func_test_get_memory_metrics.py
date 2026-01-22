from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(metricsutils.MetricsUtils, '_get_metrics')
@mock.patch.object(metricsutils.MetricsUtils, '_get_vm')
def test_get_memory_metrics(self, mock_get_vm, mock_get_metrics):
    mock_vm = mock_get_vm.return_value
    self.utils._metrics_defs_obj = {self.utils._MEMORY_METRICS: mock.sentinel.metrics}
    metrics_memory = mock.MagicMock()
    metrics_memory.MetricValue = 3
    mock_get_metrics.return_value = [metrics_memory]
    response = self.utils.get_memory_metrics(mock.sentinel.vm_name)
    self.assertEqual(3, response)
    mock_get_vm.assert_called_once_with(mock.sentinel.vm_name)
    mock_get_metrics.assert_called_once_with(mock_vm, mock.sentinel.metrics)