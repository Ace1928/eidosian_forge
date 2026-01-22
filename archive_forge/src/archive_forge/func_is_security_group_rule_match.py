from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def is_security_group_rule_match(prototype, security_group_rule):
    skip_keys = ['ether_type']
    if 'ether_type' in prototype and security_group_rule['ethertype'] != prototype['ether_type']:
        return False
    if 'protocol' in prototype and prototype['protocol'] in ['tcp', 'udp']:
        if 'port_range_max' in prototype and prototype['port_range_max'] in [-1, 65535]:
            if security_group_rule['port_range_max'] is not None:
                return False
            skip_keys.append('port_range_max')
        if 'port_range_min' in prototype and prototype['port_range_min'] in [-1, 1]:
            if security_group_rule['port_range_min'] is not None:
                return False
            skip_keys.append('port_range_min')
    if all((security_group_rule[k] == prototype[k] for k in set(prototype.keys()) - set(skip_keys))):
        return security_group_rule
    else:
        return None