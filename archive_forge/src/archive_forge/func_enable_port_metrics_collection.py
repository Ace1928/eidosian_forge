from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def enable_port_metrics_collection(self, switch_port_name):
    port = self._get_switch_port(switch_port_name)
    metrics_names = [self._NET_IN_METRICS, self._NET_OUT_METRICS]
    self._enable_metrics(port, metrics_names)