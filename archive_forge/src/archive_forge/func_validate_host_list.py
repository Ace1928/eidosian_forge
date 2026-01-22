from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_host_list(self, host_list_input):
    """ validates the host_list_input value for None and empty

        """
    try:
        for host_list in host_list_input:
            if 'host_name' in host_list.keys() and 'host_id' in host_list.keys():
                if host_list['host_name'] and host_list['host_id']:
                    errmsg = 'parameters are mutually exclusive: host_name|host_id'
                    self.module.fail_json(msg=errmsg)
            is_host_details_missing = True
            for key, value in host_list.items():
                if key == 'host_name' and (not is_none_or_empty_string(value)):
                    is_host_details_missing = False
                elif key == 'host_id' and (not is_none_or_empty_string(value)):
                    is_host_details_missing = False
            if is_host_details_missing:
                errmsg = 'Invalid input parameter for {0}'.format(key)
                self.module.fail_json(msg=errmsg)
    except Exception as e:
        errormsg = 'Failed to validate the module param with error {0}'.format(str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)