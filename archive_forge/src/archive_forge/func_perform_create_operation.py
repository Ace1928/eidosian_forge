from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def perform_create_operation(self, state, pd_details, protection_domain_name):
    """performing creation of protection domain details"""
    if state == 'present' and (not pd_details):
        self.is_id_or_new_name_in_create()
        create_change = self.create_protection_domain(protection_domain_name)
        if create_change:
            pd_details = self.get_protection_domain(protection_domain_name)
            msg = 'Protection domain created successfully, fetched protection domain details {0}'.format(str(pd_details))
            LOG.info(msg)
            return (create_change, pd_details)
    return (False, pd_details)