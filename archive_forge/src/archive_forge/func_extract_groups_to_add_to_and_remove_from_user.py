from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def extract_groups_to_add_to_and_remove_from_user(self, groups):
    groups_extract = {}
    groups_to_add = []
    groups_to_remove = []
    if isinstance(groups, list) and len(groups) > 0:
        for group in groups:
            group_name = group['name'] if isinstance(group, dict) and 'name' in group else group
            if isinstance(group, dict) and ('state' not in group or group['state'] == 'present'):
                groups_to_add.append(group_name)
            else:
                groups_to_remove.append(group_name)
    groups_extract['add'] = groups_to_add
    groups_extract['remove'] = groups_to_remove
    return groups_extract