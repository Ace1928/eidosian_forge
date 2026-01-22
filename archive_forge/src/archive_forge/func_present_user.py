from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_user(self):
    required_params = ['account', 'email', 'password', 'first_name', 'last_name']
    self.module.fail_on_missing_params(required_params=required_params)
    user = self.get_user()
    if user:
        user = self._update_user(user)
    else:
        user = self._create_user(user)
    return user