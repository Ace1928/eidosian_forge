from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def post_present(self, entity_id):
    entity = self._service.service(entity_id).get()
    self.__attach_disks(entity)
    self.__attach_nics(entity)
    self._attach_cd(entity)
    self.changed = self.__attach_numa_nodes(entity)
    self.changed = self.__attach_watchdog(entity)
    self.changed = self.__attach_graphical_console(entity)
    self.changed = self.__attach_host_devices(entity)
    self._wait_after_lease()