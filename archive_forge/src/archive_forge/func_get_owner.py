from __future__ import absolute_import, division, print_function
import json
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text, to_native
def get_owner(module):
    path = '/subscription/users/%s/owners' % module.params['username']
    resp, info = fetch_portal(module, path, 'GET')
    return json.loads(to_text(resp.read()))[0]['key']