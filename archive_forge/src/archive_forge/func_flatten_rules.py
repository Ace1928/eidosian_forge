from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_rules(d, array_index, current_value, exclude_output):
    n = 0
    result = current_value
    has_init_value = True
    if result:
        n = len(result)
    else:
        has_init_value = False
        result = []
        v = navigate_value(d, ['read', 'security_group_rules'], array_index)
        if not v:
            return current_value
        n = len(v)
    new_array_index = dict()
    if array_index:
        new_array_index.update(array_index)
    for i in range(n):
        new_array_index['read.security_group_rules'] = i
        val = dict()
        if len(result) >= i + 1 and result[i]:
            val = result[i]
        if not exclude_output:
            v = navigate_value(d, ['read', 'security_group_rules', 'description'], new_array_index)
            val['description'] = v
        if not exclude_output:
            v = navigate_value(d, ['read', 'security_group_rules', 'direction'], new_array_index)
            val['direction'] = v
        if not exclude_output:
            v = navigate_value(d, ['read', 'security_group_rules', 'ethertype'], new_array_index)
            val['ethertype'] = v
        if not exclude_output:
            v = navigate_value(d, ['read', 'security_group_rules', 'id'], new_array_index)
            val['id'] = v
        if not exclude_output:
            v = navigate_value(d, ['read', 'security_group_rules', 'port_range_max'], new_array_index)
            val['port_range_max'] = v
        if not exclude_output:
            v = navigate_value(d, ['read', 'security_group_rules', 'port_range_min'], new_array_index)
            val['port_range_min'] = v
        if not exclude_output:
            v = navigate_value(d, ['read', 'security_group_rules', 'protocol'], new_array_index)
            val['protocol'] = v
        if not exclude_output:
            v = navigate_value(d, ['read', 'security_group_rules', 'remote_address_group_id'], new_array_index)
            val['remote_address_group_id'] = v
        if not exclude_output:
            v = navigate_value(d, ['read', 'security_group_rules', 'remote_group_id'], new_array_index)
            val['remote_group_id'] = v
        if not exclude_output:
            v = navigate_value(d, ['read', 'security_group_rules', 'remote_ip_prefix'], new_array_index)
            val['remote_ip_prefix'] = v
        if len(result) >= i + 1:
            result[i] = val
        else:
            for v in val.values():
                if v is not None:
                    result.append(val)
                    break
    return result if has_init_value or result else current_value