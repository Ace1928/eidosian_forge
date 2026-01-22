from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def get_disk_latency_metrics(self, vm_name):
    metrics_latency_def = self._metrics_defs[self._DISK_LATENCY_METRICS]
    disks = self._get_vm_resources(vm_name, self._STORAGE_ALLOC_SETTING_DATA_CLASS)
    for disk in disks:
        metrics_values = self._get_metrics_values(disk, [metrics_latency_def])
        yield {'disk_latency': metrics_values[0], 'instance_id': disk.InstanceID}