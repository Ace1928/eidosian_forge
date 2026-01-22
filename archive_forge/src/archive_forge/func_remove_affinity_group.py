from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def remove_affinity_group(self):
    affinity_group = self.get_affinity_group()
    if affinity_group:
        self.result['changed'] = True
        args = {'name': self.module.params.get('name'), 'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id')}
        if not self.module.check_mode:
            res = self.query_api('deleteAffinityGroup', **args)
            poll_async = self.module.params.get('poll_async')
            if res and poll_async:
                self.poll_job(res, 'affinitygroup')
    return affinity_group