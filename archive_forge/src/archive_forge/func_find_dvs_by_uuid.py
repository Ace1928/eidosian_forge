from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, get_all_objs
def find_dvs_by_uuid(self, uuid=None):
    """Find DVS by it's UUID"""
    dvs_obj = None
    if uuid is None:
        return dvs_obj
    dvswitches = get_all_objs(self.content, [vim.DistributedVirtualSwitch])
    for dvs in dvswitches:
        if dvs.uuid == uuid:
            dvs_obj = dvs
            break
    return dvs_obj