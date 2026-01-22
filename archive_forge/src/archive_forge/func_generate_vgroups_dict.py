from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_vgroups_dict(module, array):
    vgroups_info = {}
    api_version = array._list_available_rest_versions()
    if AC_REQUIRED_API_VERSION in api_version:
        vgroups = array.list_vgroups(pending=False)
        for vgroup in range(0, len(vgroups)):
            virtgroup = vgroups[vgroup]['name']
            vgroups_info[virtgroup] = {'volumes': vgroups[vgroup]['volumes']}
    if V6_MINIMUM_API_VERSION in api_version:
        arrayv6 = get_array(module)
        vgroups = list(arrayv6.get_volume_groups(destroyed=False).items)
        for vgroup in range(0, len(vgroups)):
            name = vgroups[vgroup].name
            vgroups_info[name]['snapshots_space'] = vgroups[vgroup].space.snapshots
            vgroups_info[name]['system'] = vgroups[vgroup].space.unique
            vgroups_info[name]['unique_space'] = vgroups[vgroup].space.unique
            vgroups_info[name]['virtual_space'] = vgroups[vgroup].space.virtual
            vgroups_info[name]['data_reduction'] = vgroups[vgroup].space.data_reduction
            vgroups_info[name]['total_reduction'] = vgroups[vgroup].space.total_reduction
            vgroups_info[name]['total_provisioned'] = vgroups[vgroup].space.total_provisioned
            vgroups_info[name]['thin_provisioning'] = vgroups[vgroup].space.thin_provisioning
            vgroups_info[name]['used_provisioned'] = (getattr(vgroups[vgroup].space, 'used_provisioned', None),)
            vgroups_info[name]['bandwidth_limit'] = getattr(vgroups[vgroup].qos, 'bandwidth_limit', '')
            vgroups_info[name]['iops_limit'] = getattr(vgroups[vgroup].qos, 'iops_limit', '')
            if SUBS_API_VERSION in api_version:
                vgroups_info[name]['total_used'] = getattr(vgroups[vgroup].space, 'total_used', None)
        if SAFE_MODE_VERSION in api_version:
            for vgroup in range(0, len(vgroups)):
                name = vgroups[vgroup].name
                vgroups_info[name]['priority_adjustment'] = vgroups[vgroup].priority_adjustment.priority_adjustment_operator + str(vgroups[vgroup].priority_adjustment.priority_adjustment_value)
    return vgroups_info