from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def validate_hash_expression(self, expression):
    """ Validate a 'hash expression' alert definition
        """
    for key in ['options', 'eval_method', 'mode']:
        if key not in expression:
            msg = 'Hash expression is missing required field {key}'.format(key=key)
            self.module.fail_json(msg)