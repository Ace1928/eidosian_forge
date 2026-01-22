from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_lag_dict(blade):
    lag_info = {}
    groups = blade.link_aggregation_groups.list_link_aggregation_groups()
    for groupcnt in range(0, len(groups.items)):
        lag_name = groups.items[groupcnt].name
        lag_info[lag_name] = {'lag_speed': groups.items[groupcnt].lag_speed, 'port_speed': groups.items[groupcnt].port_speed, 'status': groups.items[groupcnt].status}
        lag_info[lag_name]['ports'] = []
        for port in range(0, len(groups.items[groupcnt].ports)):
            lag_info[lag_name]['ports'].append({'name': groups.items[groupcnt].ports[port].name})
    return lag_info