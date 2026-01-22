from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def exists_hook(self, project, hook_url):
    hook = self.find_hook(project, hook_url)
    if hook:
        self.hook_object = hook
        return True
    return False