from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def delete_protection_domain(self, protection_domain_id):
    """
        Delete Protection Domain
        :param protection_domain_id: ID of the protection domain
        :type protection_domain_id: str
        :return: Boolean indicating if delete operation is successful
        """
    try:
        self.powerflex_conn.protection_domain.delete(protection_domain_id)
        LOG.info('Protection domain deleted successfully.')
        return True
    except Exception as e:
        error_msg = "Delete protection domain '%s' operation failed with error '%s'" % (protection_domain_id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)