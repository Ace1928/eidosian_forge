from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(metricsutils.MetricsUtils, '_get_metrics')
@mock.patch.object(metricsutils.MetricsUtils, '_get_vm_resources')
@mock.patch.object(metricsutils.MetricsUtils, '_get_vm')
def test_get_cpu_metrics(self, mock_get_vm, mock_get_vm_resources, mock_get_metrics):
    fake_cpu_count = 2
    fake_uptime = 1000
    fake_cpu_metrics_val = 2000
    self.utils._metrics_defs_obj = {self.utils._CPU_METRICS: mock.sentinel.metrics}
    mock_vm = mock_get_vm.return_value
    mock_vm.OnTimeInMilliseconds = fake_uptime
    mock_cpu = mock.MagicMock(VirtualQuantity=fake_cpu_count)
    mock_get_vm_resources.return_value = [mock_cpu]
    mock_metric = mock.MagicMock(MetricValue=fake_cpu_metrics_val)
    mock_get_metrics.return_value = [mock_metric]
    cpu_metrics = self.utils.get_cpu_metrics(mock.sentinel.vm_name)
    self.assertEqual(3, len(cpu_metrics))
    self.assertEqual(fake_cpu_metrics_val, cpu_metrics[0])
    self.assertEqual(fake_cpu_count, cpu_metrics[1])
    self.assertEqual(fake_uptime, cpu_metrics[2])
    mock_get_vm.assert_called_once_with(mock.sentinel.vm_name)
    mock_get_vm_resources.assert_called_once_with(mock.sentinel.vm_name, self.utils._PROCESSOR_SETTING_DATA_CLASS)
    mock_get_metrics.assert_called_once_with(mock_vm, mock.sentinel.metrics)