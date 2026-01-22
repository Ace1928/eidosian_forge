from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def get_affinity_group(self):
    if not self.affinity_group:
        args = {'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'name': self.module.params.get('name')}
        affinity_groups = self.query_api('listAffinityGroups', **args)
        if affinity_groups:
            self.affinity_group = affinity_groups['affinitygroup'][0]
    return self.affinity_group