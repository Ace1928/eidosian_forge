from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_cg_replication_params(self, replication):
    """ Validate cg replication params """
    if replication is None:
        errormsg = 'Please specify replication_params to enable replication.'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    else:
        self.validate_destination_pool_info(replication)
        self.validate_replication_mode(replication)
        if replication['replication_type'] == 'remote' and replication['remote_system'] is None:
            errormsg = "remote_system is required together with 'remote' replication_type"
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        if 'destination_cg_name' in replication and replication['destination_cg_name'] is not None:
            dst_cg_name_length = len(replication['destination_cg_name'])
            if dst_cg_name_length == 0 or dst_cg_name_length > 95:
                errormsg = 'destination_cg_name value should be in range of 1 to 95'
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)