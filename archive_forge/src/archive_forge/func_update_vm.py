import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def update_vm(self, vm_name, memory_mb, memory_per_numa_node, vcpus_num, vcpus_per_numa_node, limit_cpu_features, dynamic_mem_ratio, configuration_root_dir=None, snapshot_dir=None, host_shutdown_action=None, vnuma_enabled=None, snapshot_type=None, is_planned_vm=False, chassis_asset_tag=None):
    vmsetting = self._lookup_vm_check(vm_name, for_update=True)
    if host_shutdown_action:
        vmsetting.AutomaticShutdownAction = host_shutdown_action
    if configuration_root_dir:
        vmsetting.ConfigurationDataRoot = configuration_root_dir
        vmsetting.LogDataRoot = configuration_root_dir
        vmsetting.SnapshotDataRoot = configuration_root_dir
        vmsetting.SuspendDataRoot = configuration_root_dir
        vmsetting.SwapFileDataRoot = configuration_root_dir
    if vnuma_enabled is not None:
        vmsetting.VirtualNumaEnabled = vnuma_enabled
    self._set_vm_memory(vmsetting, memory_mb, memory_per_numa_node, dynamic_mem_ratio)
    self._set_vm_vcpus(vmsetting, vcpus_num, vcpus_per_numa_node, limit_cpu_features)
    if snapshot_type:
        self._set_vm_snapshot_type(vmsetting, snapshot_type)
    if chassis_asset_tag:
        vmsetting.ChassisAssetTag = chassis_asset_tag
    self._modify_virtual_system(vmsetting)