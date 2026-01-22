from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_input_string(self):
    """ validates the input string checks if it is empty string

        """
    invalid_string = ''
    try:
        no_chk_list = ['snap_schedule', 'description']
        for key in self.module.params:
            val = self.module.params[key]
            if key not in no_chk_list and isinstance(val, str) and (val == invalid_string):
                errmsg = 'Invalid input parameter "" for {0}'.format(key)
                self.module.fail_json(msg=errmsg)
    except Exception as e:
        errormsg = 'Failed to validate the module param with error {0}'.format(str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)