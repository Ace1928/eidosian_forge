from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
def test_enable_metrics_exception(self):
    metric_name = self.utils._CPU_METRICS
    metric_def = mock.MagicMock()
    self.utils._metrics_defs_obj = {metric_name: metric_def}
    self.utils._metrics_svc.ControlMetrics.return_value = [1]
    self.assertRaises(exceptions.OSWinException, self.utils._enable_metrics, mock.MagicMock(), [metric_name])