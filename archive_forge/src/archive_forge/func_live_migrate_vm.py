import platform
from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import migrationutils
from os_win.utils.compute import vmutils
def live_migrate_vm(self, vm_name, dest_host, migrate_disks=True):
    self.check_live_migration_config()
    conn_v2_remote = self._get_conn_v2(dest_host)
    vm = self._get_vm(self._compat_conn, vm_name)
    rmt_ip_addr_list = self._get_ip_address_list(conn_v2_remote, dest_host)
    planned_vm = self._get_planned_vm(vm_name, conn_v2_remote)
    if migrate_disks:
        new_resource_setting_data = self._get_vhd_setting_data(vm)
        migration_type = self._MIGRATION_TYPE_VIRTUAL_SYSTEM_AND_STORAGE
    else:
        new_resource_setting_data = None
        migration_type = self._MIGRATION_TYPE_VIRTUAL_SYSTEM
    self._live_migrate_vm(self._compat_conn, vm, planned_vm, rmt_ip_addr_list, new_resource_setting_data, dest_host, migration_type)