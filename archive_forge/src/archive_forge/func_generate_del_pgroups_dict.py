from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_del_pgroups_dict(module, array):
    pgroups_info = {}
    api_version = array._list_available_rest_versions()
    pgroups = array.list_pgroups(pending_only=True)
    if SHARED_CAP_API_VERSION in api_version:
        array_v6 = get_array(module)
        deleted_enabled = True
    else:
        deleted_enabled = False
    for pgroup in range(0, len(pgroups)):
        protgroup = pgroups[pgroup]['name']
        pgroups_info[protgroup] = {'hgroups': pgroups[pgroup]['hgroups'], 'hosts': pgroups[pgroup]['hosts'], 'source': pgroups[pgroup]['source'], 'targets': pgroups[pgroup]['targets'], 'volumes': pgroups[pgroup]['volumes'], 'time_remaining': pgroups[pgroup]['time_remaining']}
        try:
            prot_sched = array.get_pgroup(protgroup, schedule=True, pending=True)
            prot_reten = array.get_pgroup(protgroup, retention=True, pending=True)
            snap_transfers = array.get_pgroup(protgroup, snap=True, transfer=True, pending=True)
        except purestorage.PureHTTPError as err:
            if err.code == 400:
                continue
        if prot_sched['snap_enabled'] or prot_sched['replicate_enabled']:
            pgroups_info[protgroup]['snap_frequency'] = prot_sched['snap_frequency']
            pgroups_info[protgroup]['replicate_frequency'] = prot_sched['replicate_frequency']
            pgroups_info[protgroup]['snap_enabled'] = prot_sched['snap_enabled']
            pgroups_info[protgroup]['replicate_enabled'] = prot_sched['replicate_enabled']
            pgroups_info[protgroup]['snap_at'] = prot_sched['snap_at']
            pgroups_info[protgroup]['replicate_at'] = prot_sched['replicate_at']
            pgroups_info[protgroup]['replicate_blackout'] = prot_sched['replicate_blackout']
            pgroups_info[protgroup]['per_day'] = prot_reten['per_day']
            pgroups_info[protgroup]['target_per_day'] = prot_reten['target_per_day']
            pgroups_info[protgroup]['target_days'] = prot_reten['target_days']
            pgroups_info[protgroup]['days'] = prot_reten['days']
            pgroups_info[protgroup]['all_for'] = prot_reten['all_for']
            pgroups_info[protgroup]['target_all_for'] = prot_reten['target_all_for']
        pgroups_info[protgroup]['snaps'] = {}
        for snap_transfer in range(0, len(snap_transfers)):
            snap = snap_transfers[snap_transfer]['name']
            pgroups_info[protgroup]['snaps'][snap] = {'time_remaining': snap_transfers[snap_transfer]['time_remaining'], 'created': snap_transfers[snap_transfer]['created'], 'started': snap_transfers[snap_transfer]['started'], 'completed': snap_transfers[snap_transfer]['completed'], 'physical_bytes_written': snap_transfers[snap_transfer]['physical_bytes_written'], 'data_transferred': snap_transfers[snap_transfer]['data_transferred'], 'progress': snap_transfers[snap_transfer]['progress']}
        if deleted_enabled:
            pgroups_info[protgroup]['deleted_volumes'] = []
            volumes = list(array_v6.get_protection_groups_volumes(group_names=[protgroup]).items)
            if volumes:
                for volume in range(0, len(volumes)):
                    if volumes[volume].member['destroyed']:
                        pgroups_info[protgroup]['deleted_volumes'].append(volumes[volume].member['name'])
            else:
                pgroups_info[protgroup]['deleted_volumes'] = None
        if PER_PG_VERSION in api_version:
            try:
                pgroups_info[protgroup]['retention_lock'] = list(array_v6.get_protection_groups(names=[protgroup]).items)[0].retention_lock
                pgroups_info[protgroup]['manual_eradication'] = list(array_v6.get_protection_groups(names=[protgroup]).items)[0].eradication_config.manual_eradication
            except Exception:
                pass
    if V6_MINIMUM_API_VERSION in api_version:
        pgroups = list(array_v6.get_protection_groups(destroyed=True).items)
        for pgroup in range(0, len(pgroups)):
            name = pgroups[pgroup].name
            pgroups_info[name]['snapshots'] = getattr(pgroups[pgroup].space, 'snapshots', None)
            pgroups_info[name]['shared'] = getattr(pgroups[pgroup].space, 'shared', None)
            pgroups_info[name]['data_reduction'] = getattr(pgroups[pgroup].space, 'data_reduction', None)
            pgroups_info[name]['thin_provisioning'] = getattr(pgroups[pgroup].space, 'thin_provisioning', None)
            pgroups_info[name]['total_physical'] = getattr(pgroups[pgroup].space, 'total_physical', None)
            pgroups_info[name]['total_provisioned'] = getattr(pgroups[pgroup].space, 'total_provisioned', None)
            pgroups_info[name]['total_reduction'] = getattr(pgroups[pgroup].space, 'total_reduction', None)
            pgroups_info[name]['unique'] = getattr(pgroups[pgroup].space, 'unique', None)
            pgroups_info[name]['virtual'] = getattr(pgroups[pgroup].space, 'virtual', None)
            pgroups_info[name]['replication'] = getattr(pgroups[pgroup].space, 'replication', None)
            pgroups_info[name]['used_provisioned'] = getattr(pgroups[pgroup].space, 'used_provisioned', None)
    return pgroups_info