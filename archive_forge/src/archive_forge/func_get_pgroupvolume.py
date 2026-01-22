from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
def get_pgroupvolume(module, array):
    """Return Protection Group Volume or None"""
    try:
        volumes = []
        pgroup = list(array.get_protection_groups(names=[module.params['name']]).items)[0]
        if pgroup.host_count > 0:
            host_dict = list(array.get_protection_groups_hosts(group_names=[module.params['name']]).items)
            for host in range(0, len(host_dict)):
                hostvols = list(array.get_connections(host_names=[host_dict[host].member['name']]).items)
                for hvol in range(0, len(hostvols)):
                    volumes.append(hostvols[hvol].volume['name'])
        elif pgroup.host_group_count > 0:
            hgroup_dict = list(array.get_protection_groups_host_groups(group_names=[module.params['name']]).items)
            hgroups = []
            for hgentry in range(0, len(hgroup_dict)):
                hgvols = list(array.get_connections(host_group_names=[hgroup_dict[hgentry].member['name']]).items)
                for hgvol in range(0, len(hgvols)):
                    volumes.append(hgvols[hgvol].volume['name'])
            for hgroup in range(0, len(hgroup_dict)):
                hg_hosts = list(array.get_host_groups_hosts(group_names=[hgroup_dict[hgroup].member['name']]).items)
                for hg_host in range(0, len(hg_hosts)):
                    host_vols = list(array.get_connections(host_names=[hg_hosts[hg_host].member['name']]).items)
                    for host_vol in range(0, len(host_vols)):
                        volumes.append(host_vols[host_vol].volume['name'])
        else:
            vol_dict = list(array.get_protection_groups_volumes(group_names=[module.params['name']]).items)
            for entry in range(0, len(vol_dict)):
                volumes.append(vol_dict[entry].member['name'])
        volumes = list(set(volumes))
        if '::' in module.params['name']:
            restore_volume = module.params['name'].split('::')[0] + '::' + module.params['restore']
        else:
            restore_volume = module.params['restore']
        for volume in volumes:
            if volume == restore_volume:
                return volume
    except Exception:
        return None