from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def get_cpu_metrics(self, vm_name):
    vm = self._get_vm(vm_name)
    cpu_sd = self._get_vm_resources(vm_name, self._PROCESSOR_SETTING_DATA_CLASS)[0]
    cpu_metrics_def = self._metrics_defs[self._CPU_METRICS]
    cpu_metrics_aggr = self._get_metrics(vm, cpu_metrics_def)
    cpu_used = 0
    if cpu_metrics_aggr:
        cpu_used = int(cpu_metrics_aggr[0].MetricValue)
    return (cpu_used, int(cpu_sd.VirtualQuantity), int(vm.OnTimeInMilliseconds))