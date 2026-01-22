from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_del_snap_dict(module, array):
    snap_info = {}
    api_version = array._list_available_rest_versions()
    if FC_REPL_API_VERSION in api_version:
        arrayv6 = get_array(module)
        snapsv6 = list(arrayv6.get_volume_snapshots(destroyed=True).items)
    snaps = array.list_volumes(snap=True, pending_only=True)
    for snap in range(0, len(snaps)):
        snapshot = snaps[snap]['name']
        snap_info[snapshot] = {'size': snaps[snap]['size'], 'source': snaps[snap]['source'], 'created': snaps[snap]['created'], 'time_remaining': snaps[snap]['time_remaining'], 'tags': [], 'remote': []}
    if FC_REPL_API_VERSION in api_version:
        for snap in range(0, len(snapsv6)):
            snapshot = snapsv6[snap].name
            snap_info[snapshot]['snapshot_space'] = snapsv6[snap].space.snapshots
            snap_info[snapshot]['used_provisioned'] = (getattr(snapsv6[snap].space, 'used_provisioned', None),)
            snap_info[snapshot]['total_physical'] = snapsv6[snap].space.total_physical
            snap_info[snapshot]['total_provisioned'] = snapsv6[snap].space.total_provisioned
            snap_info[snapshot]['unique_space'] = snapsv6[snap].space.unique
            if SUBS_API_VERSION in api_version:
                snap_info[snapshot]['total_used'] = snapsv6[snap].space.total_used
        offloads = list(arrayv6.get_offloads().items)
        for offload in range(0, len(offloads)):
            offload_name = offloads[offload].name
            check_offload = arrayv6.get_remote_volume_snapshots(on=offload_name)
            if check_offload.status_code == 200:
                remote_snaps = list(arrayv6.get_remote_volume_snapshots(on=offload_name, destroyed=True).items)
                for remote_snap in range(0, len(remote_snaps)):
                    remote_snap_name = remote_snaps[remote_snap].name.split(':')[1]
                    remote_transfer = list(arrayv6.get_remote_volume_snapshots_transfer(on=offload_name, names=[remote_snaps[remote_snap].name]).items)[0]
                    remote_dict = {'source': remote_snaps[remote_snap].source.name, 'suffix': remote_snaps[remote_snap].suffix, 'size': remote_snaps[remote_snap].provisioned, 'data_transferred': remote_transfer.data_transferred, 'completed': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(remote_transfer.completed / 1000)) + ' UTC', 'physical_bytes_written': remote_transfer.physical_bytes_written, 'progress': remote_transfer.progress, 'created': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(remote_snaps[remote_snap].created / 1000)) + ' UTC'}
                    try:
                        snap_info[remote_snap_name]['remote'].append(remote_dict)
                    except KeyError:
                        snap_info[remote_snap_name] = {'remote': []}
                        snap_info[remote_snap_name]['remote'].append(remote_dict)
    if ACTIVE_DR_API in api_version:
        snaptags = array.list_volumes(snap=True, tags=True, pending_only=True, namespace='*')
        for snaptag in range(0, len(snaptags)):
            if snaptags[snaptag]['namespace'] != 'vasa-integration.purestorage.com':
                snapname = snaptags[snaptag]['name']
                tagdict = {'key': snaptags[snaptag]['key'], 'value': snaptags[snaptag]['value'], 'namespace': snaptags[snaptag]['namespace']}
                snap_info[snapname]['tags'].append(tagdict)
    return snap_info