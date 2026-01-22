from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def get_memory_metrics(self, vm_name):
    vm = self._get_vm(vm_name)
    memory_def = self._metrics_defs[self._MEMORY_METRICS]
    metrics_memory = self._get_metrics(vm, memory_def)
    memory_usage = 0
    if metrics_memory:
        memory_usage = int(metrics_memory[0].MetricValue)
    return memory_usage