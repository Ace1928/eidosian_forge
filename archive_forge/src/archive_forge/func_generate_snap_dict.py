from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_snap_dict(blade):
    snap_info = {}
    snaps = blade.file_system_snapshots.list_file_system_snapshots()
    api_version = blade.api_version.list_versions().versions
    for snap in range(0, len(snaps.items)):
        snapshot = snaps.items[snap].name
        snap_info[snapshot] = {'destroyed': snaps.items[snap].destroyed, 'source': snaps.items[snap].source, 'suffix': snaps.items[snap].suffix, 'source_destroyed': snaps.items[snap].source_destroyed}
        if REPLICATION_API_VERSION in api_version:
            snap_info[snapshot]['owner'] = snaps.items[snap].owner.name
            snap_info[snapshot]['owner_destroyed'] = snaps.items[snap].owner_destroyed
            snap_info[snapshot]['source_display_name'] = snaps.items[snap].source_display_name
            snap_info[snapshot]['source_is_local'] = snaps.items[snap].source_is_local
            snap_info[snapshot]['source_location'] = snaps.items[snap].source_location.name
            snap_info[snapshot]['policies'] = []
            if PUBLIC_API_VERSION in api_version:
                for policy in range(0, len(snaps.items[snap].policies)):
                    snap_info[snapshot]['policies'].append({'name': snaps.items[snap].policies[policy].name, 'location': snaps.items[snap].policies[policy].location.name})
    return snap_info