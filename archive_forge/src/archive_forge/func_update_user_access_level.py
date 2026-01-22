from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def update_user_access_level(self, members, gitlab_user_id, access_level):
    for member in members:
        if member.id == gitlab_user_id:
            member.access_level = access_level
            member.save()