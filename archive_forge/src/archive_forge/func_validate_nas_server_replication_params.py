from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_nas_server_replication_params(self, replication):
    """ Validate NAS server replication params
            :param: replication: Dict which has all the replication parameter values
        """
    if replication is None:
        errormsg = 'Please specify replication_params to enable replication.'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    else:
        if replication['destination_pool_id'] is not None and replication['destination_pool_name'] is not None:
            errormsg = "'destination_pool_id' and 'destination_pool_name' is mutually exclusive."
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        self.validate_rpo(replication)
        if replication['replication_type'] == 'remote' and replication['remote_system'] is None:
            errormsg = "Remote_system is required together with 'remote' replication_type"
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        if 'destination_nas_name' in replication and replication['destination_nas_name'] is not None:
            dst_nas_server_name_length = len(replication['destination_nas_name'])
            if dst_nas_server_name_length == 0 or dst_nas_server_name_length > 95:
                errormsg = 'destination_nas_name value should be in range of 1 to 95'
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)