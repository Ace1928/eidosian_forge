from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rundeck import (
def get_project_facts(self):
    resp, info = api_request(module=self.module, endpoint='project/%s' % self.module.params['name'])
    return resp