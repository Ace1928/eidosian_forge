from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def get_acceleration_pool(self, acceleration_pool_name=None, acceleration_pool_id=None, protection_domain_id=None):
    """Get acceleration pool details
            :param acceleration_pool_name: Name of the acceleration pool
            :param acceleration_pool_id: ID of the acceleration pool
            :param protection_domain_id: ID of the protection domain
            :return: Acceleration pool details
            :rtype: dict
        """
    name_or_id = acceleration_pool_id if acceleration_pool_id else acceleration_pool_name
    try:
        acceleration_pool_details = None
        if acceleration_pool_id:
            acceleration_pool_details = self.powerflex_conn.acceleration_pool.get(filter_fields={'id': acceleration_pool_id})
        if acceleration_pool_name:
            acceleration_pool_details = self.powerflex_conn.acceleration_pool.get(filter_fields={'name': acceleration_pool_name, 'protectionDomainId': protection_domain_id})
        if not acceleration_pool_details:
            error_msg = "Unable to find the acceleration pool with '%s'. Please enter a valid acceleration pool name/id." % name_or_id
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
        return acceleration_pool_details[0]
    except Exception as e:
        error_msg = "Failed to get the acceleration pool '%s' with error '%s'" % (name_or_id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)