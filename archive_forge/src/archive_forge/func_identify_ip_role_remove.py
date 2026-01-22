from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def identify_ip_role_remove(self, sds_ip_list, sds_details, sds_ip_state):
    existing_ip_role_list = sds_details['ipList']
    if sds_ip_state == 'absent-in-sds':
        ips_to_remove = [ip for ip in existing_ip_role_list if ip in sds_ip_list]
        if len(ips_to_remove) != 0:
            LOG.info('IP(s) to remove: %s', ips_to_remove)
            return ips_to_remove
        else:
            LOG.info('IP(s) do not exists.')
            return []