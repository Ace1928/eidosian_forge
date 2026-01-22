from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def create_affinity_group(self):
    affinity_group = self.get_affinity_group()
    if not affinity_group:
        self.result['changed'] = True
        args = {'name': self.module.params.get('name'), 'type': self.get_affinity_type(), 'description': self.module.params.get('description'), 'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id')}
        if not self.module.check_mode:
            res = self.query_api('createAffinityGroup', **args)
            poll_async = self.module.params.get('poll_async')
            if res and poll_async:
                affinity_group = self.poll_job(res, 'affinitygroup')
    return affinity_group