from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_list_resp_security_group_rules(value):
    if not value:
        return None
    result = []
    for item in value:
        val = dict()
        val['description'] = item.get('description')
        val['direction'] = item.get('direction')
        val['ethertype'] = item.get('ethertype')
        val['id'] = item.get('id')
        val['port_range_max'] = item.get('port_range_max')
        val['port_range_min'] = item.get('port_range_min')
        val['protocol'] = item.get('protocol')
        val['remote_address_group_id'] = item.get('remote_address_group_id')
        val['remote_group_id'] = item.get('remote_group_id')
        val['remote_ip_prefix'] = item.get('remote_ip_prefix')
        val['security_group_id'] = item.get('security_group_id')
        result.append(val)
    return result