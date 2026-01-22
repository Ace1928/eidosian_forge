from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
def stop_vm(self, vm, force, timeout):
    vmid = vm['vmid']
    proxmox_node = self.proxmox_api.nodes(vm['node'])
    taskid = proxmox_node.qemu(vmid).status.shutdown.post(forceStop=1 if force else 0, timeout=timeout)
    if not self.wait_for_task(vm['node'], taskid):
        self.module.fail_json(msg='Reached timeout while waiting for stopping VM. Last line in task before timeout: %s' % proxmox_node.tasks(taskid).log.get()[:1])
        return False
    return True