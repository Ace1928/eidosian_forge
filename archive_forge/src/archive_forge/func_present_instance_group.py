from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def present_instance_group(self):
    instance_group = self.get_instance_group()
    if not instance_group:
        self.result['changed'] = True
        args = {'name': self.module.params.get('name'), 'account': self.get_account('name'), 'domainid': self.get_domain('id'), 'projectid': self.get_project('id')}
        if not self.module.check_mode:
            res = self.query_api('createInstanceGroup', **args)
            instance_group = res['instancegroup']
    return instance_group