from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(metricsutils.MetricsUtils, '_enable_metrics')
@mock.patch.object(metricsutils.MetricsUtils, '_get_switch_port')
def test_enable_switch_port_metrics_collection(self, mock_get_port, mock_enable_metrics):
    self.utils.enable_port_metrics_collection(mock.sentinel.port_name)
    mock_get_port.assert_called_once_with(mock.sentinel.port_name)
    metrics = [self.utils._NET_IN_METRICS, self.utils._NET_OUT_METRICS]
    mock_enable_metrics.assert_called_once_with(mock_get_port.return_value, metrics)