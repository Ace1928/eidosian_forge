from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
def vmstatus(self, vm, vmid):
    return getattr(self.proxmox_api.nodes(vm['node']), vm['type'])(vmid).status