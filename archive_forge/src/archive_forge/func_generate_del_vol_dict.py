from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_del_vol_dict(module, array):
    volume_info = {}
    api_version = array._list_available_rest_versions()
    vols = array.list_volumes(pending_only=True)
    for vol in range(0, len(vols)):
        volume = vols[vol]['name']
        volume_info[volume] = {'size': vols[vol]['size'], 'source': vols[vol]['source'], 'created': vols[vol]['created'], 'serial': vols[vol]['serial'], 'page83_naa': PURE_OUI + vols[vol]['serial'], 'nvme_nguid': 'eui.00' + vols[vol]['serial'][0:14].lower() + '24a937' + vols[vol]['serial'][-10:].lower(), 'time_remaining': vols[vol]['time_remaining'], 'tags': []}
    if V6_MINIMUM_API_VERSION in api_version:
        arrayv6 = get_array(module)
        vols_space = list(arrayv6.get_volumes_space(destroyed=True).items)
        for vol in range(0, len(vols_space)):
            name = vols_space[vol].name
            volume_info[name]['snapshots_space'] = vols_space[vol].space.snapshots
            volume_info[name]['system'] = vols_space[vol].space.unique
            volume_info[name]['unique_space'] = vols_space[vol].space.unique
            volume_info[name]['virtual_space'] = vols_space[vol].space.virtual
            volume_info[name]['total_physical_space'] = vols_space[vol].space.total_physical
            volume_info[name]['data_reduction'] = vols_space[vol].space.data_reduction
            volume_info[name]['total_reduction'] = vols_space[vol].space.total_reduction
            volume_info[name]['total_provisioned'] = vols_space[vol].space.total_provisioned
            volume_info[name]['thin_provisioning'] = vols_space[vol].space.thin_provisioning
            if SHARED_CAP_API_VERSION in api_version:
                volume_info[name]['snapshots_effective'] = getattr(vols_space[vol].space, 'snapshots_effective', None)
                volume_info[name]['unique_effective'] = getattr(vols_space[vol].space, 'unique_effective', None)
                volume_info[name]['used_provisioned'] = (getattr(vols_space[vol].space, 'used_provisioned', None),)
            if SUBS_API_VERSION in api_version:
                volume_info[name]['total_used'] = vols_space[vol].space.total_used
    if ACTIVE_DR_API in api_version:
        voltags = array.list_volumes(tags=True, pending_only=True)
        for voltag in range(0, len(voltags)):
            if voltags[voltag]['namespace'] != 'vasa-integration.purestorage.com':
                volume = voltags[voltag]['name']
                tagdict = {'key': voltags[voltag]['key'], 'value': voltags[voltag]['value'], 'copyable': voltags[voltag]['copyable'], 'namespace': voltags[voltag]['namespace']}
                volume_info[volume]['tags'].append(tagdict)
    if V6_MINIMUM_API_VERSION in api_version:
        volumes = list(arrayv6.get_volumes(destroyed=True).items)
        for vol in range(0, len(volumes)):
            name = volumes[vol].name
            volume_info[name]['promotion_status'] = volumes[vol].promotion_status
            volume_info[name]['requested_promotion_state'] = volumes[vol].requested_promotion_state
            if SAFE_MODE_VERSION in api_version:
                volume_info[name]['subtype'] = volumes[vol].subtype
                volume_info[name]['priority'] = volumes[vol].priority
                volume_info[name]['priority_adjustment'] = volumes[vol].priority_adjustment.priority_adjustment_operator + str(volumes[vol].priority_adjustment.priority_adjustment_value)
    return volume_info