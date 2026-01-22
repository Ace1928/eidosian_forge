import platform
from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import migrationutils
from os_win.utils.compute import vmutils
def create_planned_vm(self, vm_name, src_host, disk_path_mapping):
    dest_host = platform.node()
    vmutils_remote = vmutils.VMUtils(src_host)
    conn_v2_remote = self._get_conn_v2(src_host)
    vm = self._get_vm(conn_v2_remote, vm_name)
    self.destroy_existing_planned_vm(vm_name)
    ip_addr_list = self._get_ip_address_list(self._compat_conn, dest_host)
    disk_paths = self._get_disk_data(vm_name, vmutils_remote, disk_path_mapping)
    planned_vm = self._create_planned_vm(self._compat_conn, conn_v2_remote, vm, ip_addr_list, dest_host)
    self._update_planned_vm_disk_resources(self._compat_conn, planned_vm, vm_name, disk_paths)