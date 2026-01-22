from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible.module_utils._text import to_native
def create_fcd_result(self, state):
    result = dict(name=self.disk.config.name, datastore_name=self.disk.config.backing.datastore.name, size_mb=self.disk.config.capacityInMB, state=state)
    return result