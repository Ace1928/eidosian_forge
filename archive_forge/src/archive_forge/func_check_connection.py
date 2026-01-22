from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, url_argument_spec
def check_connection(self):
    ret = self.call_url('v1/status')
    if ret['code'] == 200:
        return True
    return False