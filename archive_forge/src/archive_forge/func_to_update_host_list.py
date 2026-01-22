from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def to_update_host_list(snapshot, host, host_state):
    """ Determines whether to update hosts list or not"""
    hosts_dict = get_hosts_dict(snapshot)
    if (not hosts_dict or host not in list(hosts_dict.keys())) and host_state == 'mapped':
        return True
    if (hosts_dict and host in list(hosts_dict.keys())) and host_state == 'unmapped':
        return True
    return False