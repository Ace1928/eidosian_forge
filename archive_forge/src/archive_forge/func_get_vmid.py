from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_vmid(self, name, ignore_missing=False, choose_first_if_multiple=False):
    try:
        vms = [vm['vmid'] for vm in self.proxmox_api.cluster.resources.get(type='vm') if vm.get('name') == name]
    except Exception as e:
        self.module.fail_json(msg='Unable to retrieve list of VMs filtered by name %s: %s' % (name, e))
    if not vms:
        if ignore_missing:
            return None
        self.module.fail_json(msg='No VM with name %s found' % name)
    elif len(vms) > 1:
        self.module.fail_json(msg='Multiple VMs with name %s found, provide vmid instead' % name)
    return vms[0]