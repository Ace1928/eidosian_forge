from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def query_resource_profiles(self):
    """ Returns a set of the profile objects objects assigned to the resource
        """
    url = '{resource_url}/policy_profiles?expand=resources'
    try:
        response = self.client.get(url.format(resource_url=self.resource_url))
    except Exception as e:
        msg = 'Failed to query {resource_type} policies: {error}'.format(resource_type=self.resource_type, error=e)
        self.module.fail_json(msg=msg)
    resources = response.get('resources', [])
    profiles = [self.clean_profile_object(profile) for profile in resources]
    return profiles