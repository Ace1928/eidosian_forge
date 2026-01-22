from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.memset import get_zone_id
from ansible_collections.community.general.plugins.module_utils.memset import memset_api_call

    We need to perform some initial sanity checking and also look
    up required info before handing it off to create or delete functions.
    Check mode is integrated into the create or delete functions.
    