from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def ensure_key_absent(session, name, check_mode):
    to_delete = [key for key in get_all_keys(session) if key['title'] == name]
    delete_keys(session, to_delete, check_mode=check_mode)
    return {'changed': bool(to_delete), 'deleted_keys': to_delete}