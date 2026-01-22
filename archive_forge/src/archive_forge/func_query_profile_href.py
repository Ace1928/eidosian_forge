from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def query_profile_href(self, profile):
    """ Add or Update the policy_profile href field

        Example:
            {name: STR, ...} => {name: STR, href: STR}
        """
    resource = self.manageiq.find_collection_resource_or_fail('policy_profiles', **profile)
    return dict(name=profile['name'], href=resource['href'])