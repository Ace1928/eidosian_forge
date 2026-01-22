from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def identify_ip_role_add(self, sds_ip_list, sds_details, sds_ip_state):
    existing_ip_role_list = sds_details['ipList']
    update_role = []
    ips_to_add = []
    existing_ip_list = []
    if existing_ip_role_list:
        for ip in existing_ip_role_list:
            existing_ip_list.append(ip['ip'])
    for given_ip in sds_ip_list:
        ip = given_ip['ip']
        if ip not in existing_ip_list:
            ips_to_add.append(given_ip)
    LOG.info('IP(s) to be added: %s', ips_to_add)
    if len(ips_to_add) != 0:
        for ip in ips_to_add:
            sds_ip_list.remove(ip)
    update_role = [ip for ip in sds_ip_list if ip not in existing_ip_role_list]
    LOG.info('Role update needed for: %s', update_role)
    return (ips_to_add, update_role)