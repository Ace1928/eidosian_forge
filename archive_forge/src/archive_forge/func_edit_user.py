from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def edit_user(self, user, name, group, password, email):
    """ Edit a user from manageiq.

        Returns:
            a short message describing the operation executed.
        """
    group_id = None
    url = '%s/users/%s' % (self.api_url, user['id'])
    resource = dict(userid=user['userid'])
    if group is not None:
        group_id = self.group_id(group)
        resource['group'] = dict(id=group_id)
    if name is not None:
        resource['name'] = name
    if email is not None:
        resource['email'] = email
    if self.module.params['update_password'] == 'on_create':
        password = None
    if password is not None:
        resource['password'] = password
    if self.compare_user(user, name, group_id, password, email):
        return dict(changed=False, msg='user %s is not changed.' % user['userid'])
    try:
        result = self.client.post(url, action='edit', resource=resource)
    except Exception as e:
        self.module.fail_json(msg='failed to update user %s: %s' % (user['userid'], str(e)))
    return dict(changed=True, msg='successfully updated the user %s: %s' % (user['userid'], result))