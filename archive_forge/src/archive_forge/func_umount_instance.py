from __future__ import absolute_import, division, print_function
import re
import time
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.proxmox import (
def umount_instance(self, vm, vmid, timeout):
    taskid = getattr(self.proxmox_api.nodes(vm['node']), VZ_TYPE)(vmid).status.umount.post()
    while timeout:
        if self.api_task_ok(vm['node'], taskid):
            return True
        timeout -= 1
        if timeout == 0:
            self.module.fail_json(vmid=vmid, taskid=taskid, msg='Reached timeout while waiting for unmounting VM. Last line in task before timeout: %s' % self.proxmox_api.nodes(vm['node']).tasks(taskid).log.get()[:1])
        time.sleep(1)
    return False