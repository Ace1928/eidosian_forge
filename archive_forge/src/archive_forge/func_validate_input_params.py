from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def validate_input_params(self):
    """Validate the input parameters"""
    name_params = ['protection_domain_name', 'protection_domain_new_name', 'protection_domain_id']
    msg = 'Please provide the valid {0}'
    for n_item in name_params:
        if self.module.params[n_item] is not None and (len(self.module.params[n_item].strip()) or self.module.params[n_item].count(' ') > 0) == 0:
            err_msg = msg.format(n_item)
            self.module.fail_json(msg=err_msg)
    if self.module.params['network_limits'] is not None:
        if self.module.params['network_limits']['overall_limit'] is not None and self.module.params['network_limits']['overall_limit'] < 0:
            error_msg = 'Overall limit cannot be negative. Provide a valid value '
            LOG.info(error_msg)
            self.module.fail_json(msg=error_msg)