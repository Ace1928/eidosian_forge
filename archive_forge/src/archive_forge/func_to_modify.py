from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def to_modify(self, sds_details, sds_new_name, rfcache_enabled, rmcache_enabled, rmcache_size, performance_profile):
    """
        :param sds_details: Details of the SDS
        :type sds_details: dict
        :param sds_new_name: New name of SDS
        :type sds_new_name: str
        :param rfcache_enabled: Whether to enable the Read Flash cache
        :type rfcache_enabled: bool
        :param rmcache_enabled: Whether to enable the Read RAM cache
        :type rmcache_enabled: bool
        :param rmcache_size: Read RAM cache size (in MB)
        :type rmcache_size: int
        :param performance_profile: Performance profile to apply to the SDS
        :type performance_profile: str
        :return: Dictionary containing the attributes of SDS which are to be
                 updated
        :rtype: dict
        """
    modify_dict = {}
    if sds_new_name is not None and sds_new_name != sds_details['name']:
        modify_dict['name'] = sds_new_name
    param_input = dict()
    param_input['rfcacheEnabled'] = rfcache_enabled
    param_input['rmcacheEnabled'] = rmcache_enabled
    param_input['perfProfile'] = performance_profile
    param_list = ['rfcacheEnabled', 'rmcacheEnabled', 'perfProfile']
    for param in param_list:
        if param_input[param] is not None and sds_details[param] != param_input[param]:
            modify_dict[param] = param_input[param]
    if rmcache_size is not None:
        self.validate_rmcache_size_parameter(rmcache_enabled, rmcache_size)
        exisitng_size_mb = sds_details['rmcacheSizeInKb'] / 1024
        if rmcache_size != exisitng_size_mb:
            if sds_details['rmcacheEnabled']:
                modify_dict['rmcacheSizeInMB'] = rmcache_size
            else:
                error_msg = "Failed to update RM cache size for the SDS '%s' as RM cache is disabled previously, please enable it before setting the size." % sds_details['name']
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
    return modify_dict