from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def get_members_in_a_group(self, gitlab_group_id):
    group = self._gitlab.groups.get(gitlab_group_id)
    return group.members.list(all=True)