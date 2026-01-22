from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def assign_or_unassign_profiles(self, profiles, action):
    """ Perform assign/unassign action
        """
    profiles_to_post = self.profiles_to_update(profiles, action)
    if not profiles_to_post:
        return dict(changed=False, msg='Profiles {profiles} already {action}ed, nothing to do'.format(action=action, profiles=profiles))
    url = '{resource_url}/policy_profiles'.format(resource_url=self.resource_url)
    try:
        response = self.client.post(url, action=action, resources=profiles_to_post)
    except Exception as e:
        msg = 'Failed to {action} profile: {error}'.format(action=action, error=e)
        self.module.fail_json(msg=msg)
    for result in response['results']:
        if not result['success']:
            msg = 'Failed to {action}: {message}'.format(action=action, message=result['message'])
            self.module.fail_json(msg=msg)
    return dict(changed=True, msg='Successfully {action}ed profiles: {profiles}'.format(action=action, profiles=profiles))