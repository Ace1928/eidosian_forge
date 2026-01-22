from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
def test_sum_metrics_values(self):
    mock_metric = mock.MagicMock(MetricValue='100')
    result = self.utils._sum_metrics_values([mock_metric] * 2)
    self.assertEqual(200, result)