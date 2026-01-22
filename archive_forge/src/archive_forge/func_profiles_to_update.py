from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def profiles_to_update(self, profiles, action):
    """ Create a list of policies we need to update in ManageIQ.

        Returns:
            Whether or not a change took place and a message describing the
            operation executed.
        """
    profiles_to_post = []
    assigned_profiles = self.query_resource_profiles()
    assigned_profiles_set = set([profile['profile_name'] for profile in assigned_profiles])
    for profile in profiles:
        assigned = profile.get('name') in assigned_profiles_set
        if action == 'unassign' and assigned or (action == 'assign' and (not assigned)):
            profile = self.query_profile_href(profile)
            profiles_to_post.append(profile)
    return profiles_to_post