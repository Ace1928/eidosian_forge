from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def assign_user_to_group(self, user, group_identifier, access_level):
    group = find_group(self._gitlab, group_identifier)
    if self._module.check_mode:
        return True
    if group is None:
        return False
    if self.member_exists(group, self.get_user_id(user)):
        member = self.find_member(group, self.get_user_id(user))
        if not self.member_as_good_access_level(group, member.id, self.ACCESS_LEVEL[access_level]):
            member.access_level = self.ACCESS_LEVEL[access_level]
            member.save()
            return True
    else:
        try:
            group.members.create({'user_id': self.get_user_id(user), 'access_level': self.ACCESS_LEVEL[access_level]})
        except gitlab.exceptions.GitlabCreateError as e:
            self._module.fail_json(msg='Failed to assign user to group: %s' % to_native(e))
        return True
    return False