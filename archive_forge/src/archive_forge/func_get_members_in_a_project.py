from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def get_members_in_a_project(self, gitlab_project_id):
    project = self._gitlab.projects.get(gitlab_project_id)
    return project.members.list(all=True)