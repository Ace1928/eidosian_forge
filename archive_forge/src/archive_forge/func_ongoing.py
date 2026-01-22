from __future__ import absolute_import, division, print_function
import datetime
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def ongoing(self, http_call=fetch_url):
    url = 'https://api.pagerduty.com/maintenance_windows?filter=ongoing'
    headers = dict(self.headers)
    response, info = http_call(self.module, url, headers=headers)
    if info['status'] != 200:
        self.module.fail_json(msg='failed to lookup the ongoing window: %s' % info['msg'])
    json_out = self._read_response(response)
    return (False, json_out, False)