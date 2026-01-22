from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_resource_state(self, resource):
    resource_cfg = self.find_netioc_by_key(resource['name'])
    if resource_cfg is None:
        self.module.fail_json(msg="NetIOC resource named '%s' was not found" % resource['name'])
    rc = {'limit': resource_cfg.allocationInfo.limit, 'shares_level': resource_cfg.allocationInfo.shares.level}
    if resource_cfg.allocationInfo.shares.level == 'custom':
        rc['shares'] = resource_cfg.allocationInfo.shares.shares
    if self.dvs.config.networkResourceControlVersion == 'version3':
        rc['reservation'] = resource_cfg.allocationInfo.reservation
    for k, v in rc.items():
        if k in resource and v != resource[k]:
            return 'update'
    return 'valid'