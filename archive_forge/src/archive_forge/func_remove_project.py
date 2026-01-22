from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rundeck import (
def remove_project(self):
    facts = self.get_project_facts()
    if facts is None:
        self.module.exit_json(changed=False, before={}, after={})
    else:
        if not self.module.check_mode:
            api_request(module=self.module, endpoint='project/%s' % self.module.params['name'], method='DELETE')
        self.module.exit_json(changed=True, before=facts, after={})