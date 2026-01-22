from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_hgroups_dict(module, array):
    hgroups_info = {}
    api_version = array._list_available_rest_versions()
    hgroups = array.list_hgroups()
    for hgroup in range(0, len(hgroups)):
        hostgroup = hgroups[hgroup]['name']
        hgroups_info[hostgroup] = {'hosts': hgroups[hgroup]['hosts'], 'pgs': [], 'vols': []}
    pghgroups = array.list_hgroups(protect=True)
    for pghg in range(0, len(pghgroups)):
        pgname = pghgroups[pghg]['name']
        hgroups_info[pgname]['pgs'].append(pghgroups[pghg]['protection_group'])
    volhgroups = array.list_hgroups(connect=True)
    for pgvol in range(0, len(volhgroups)):
        pgname = volhgroups[pgvol]['name']
        volpgdict = [volhgroups[pgvol]['vol'], volhgroups[pgvol]['lun']]
        hgroups_info[pgname]['vols'].append(volpgdict)
    if V6_MINIMUM_API_VERSION in api_version:
        arrayv6 = get_array(module)
        hgroups = list(arrayv6.get_host_groups().items)
        for hgroup in range(0, len(hgroups)):
            if hgroups[hgroup].is_local:
                name = hgroups[hgroup].name
                hgroups_info[name]['snapshots'] = getattr(hgroups[hgroup].space, 'snapshots', None)
                hgroups_info[name]['data_reduction'] = getattr(hgroups[hgroup].space, 'data_reduction', None)
                hgroups_info[name]['thin_provisioning'] = getattr(hgroups[hgroup].space, 'thin_provisioning', None)
                hgroups_info[name]['total_physical'] = getattr(hgroups[hgroup].space, 'total_physical', None)
                hgroups_info[name]['total_provisioned'] = getattr(hgroups[hgroup].space, 'total_provisioned', None)
                hgroups_info[name]['total_reduction'] = getattr(hgroups[hgroup].space, 'total_reduction', None)
                hgroups_info[name]['unique'] = getattr(hgroups[hgroup].space, 'unique', None)
                hgroups_info[name]['virtual'] = getattr(hgroups[hgroup].space, 'virtual', None)
                hgroups_info[name]['used_provisioned'] = getattr(hgroups[hgroup].space, 'used_provisioned', None)
                if SUBS_API_VERSION in api_version:
                    hgroups_info[name]['total_used'] = hgroups[hgroup].space.total_used
    return hgroups_info