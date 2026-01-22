from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(metricsutils.MetricsUtils, '_sum_metrics_values_by_defs')
def test_get_metrics_values(self, mock_sum_by_defs):
    mock_element = mock.MagicMock()
    self.utils._conn.Msvm_MetricForME.return_value = [mock.Mock(Dependent=mock.sentinel.metric), mock.Mock(Dependent=mock.sentinel.another_metric)]
    resulted_metrics_sum = self.utils._get_metrics_values(mock_element, mock.sentinel.metrics_defs)
    self.utils._conn.Msvm_MetricForME.assert_called_once_with(Antecedent=mock_element.path_.return_value)
    mock_sum_by_defs.assert_called_once_with([mock.sentinel.metric, mock.sentinel.another_metric], mock.sentinel.metrics_defs)
    expected_metrics_sum = mock_sum_by_defs.return_value
    self.assertEqual(expected_metrics_sum, resulted_metrics_sum)