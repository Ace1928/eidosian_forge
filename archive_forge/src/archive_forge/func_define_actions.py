from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def define_actions(self, current):
    cd_action = self.na_helper.get_cd_action(current, self.parameters)
    modify = self.na_helper.get_modified_attributes(current, self.parameters) if cd_action is None else None
    if self.use_rest and cd_action is None and current and ('lock_user' not in current) and (self.parameters.get('lock_user') is not None):
        if self.parameters.get('set_password') is None:
            self.module.fail_json(msg='Error: cannot modify lock state if password is not set.')
        modify['lock_user'] = self.parameters['lock_user']
        self.na_helper.changed = True
    self.validate_action(cd_action)
    return (cd_action, modify)