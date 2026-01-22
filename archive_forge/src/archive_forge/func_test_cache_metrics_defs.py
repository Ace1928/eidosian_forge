from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
def test_cache_metrics_defs(self):
    mock_metric_def = mock.Mock(ElementName=mock.sentinel.elementname)
    self.utils._conn.CIM_BaseMetricDefinition.return_value = [mock_metric_def]
    self.utils._cache_metrics_defs()
    expected_cache_metrics = {mock.sentinel.elementname: mock_metric_def}
    self.assertEqual(expected_cache_metrics, self.utils._metrics_defs_obj)