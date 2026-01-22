from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def find_dvs_uplink_pg(self):
    dvs_uplink_pg = self.dv_switch.config.uplinkPortgroup[0] if len(self.dv_switch.config.uplinkPortgroup) else None
    return dvs_uplink_pg