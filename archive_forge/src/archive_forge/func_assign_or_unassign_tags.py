from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def assign_or_unassign_tags(self, tags, action):
    """ Perform assign/unassign action
        """
    tags_to_post = self.tags_to_update(tags, action)
    if not tags_to_post:
        return dict(changed=False, msg='Tags already {action}ed, nothing to do'.format(action=action))
    url = '{resource_url}/tags'.format(resource_url=self.resource_url)
    try:
        response = self.client.post(url, action=action, resources=tags)
    except Exception as e:
        msg = 'Failed to {action} tag: {error}'.format(action=action, error=e)
        self.module.fail_json(msg=msg)
    for result in response['results']:
        if not result['success']:
            msg = 'Failed to {action}: {message}'.format(action=action, message=result['message'])
            self.module.fail_json(msg=msg)
    return dict(changed=True, msg='Successfully {action}ed tags'.format(action=action))