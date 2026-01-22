from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def create_protected_branch(self, name, merge_access_levels, push_access_level):
    if self._module.check_mode:
        return True
    merge = self.ACCESS_LEVEL[merge_access_levels]
    push = self.ACCESS_LEVEL[push_access_level]
    self.project.protectedbranches.create({'name': name, 'merge_access_level': merge, 'push_access_level': push})