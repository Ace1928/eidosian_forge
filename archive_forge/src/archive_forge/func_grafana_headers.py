from __future__ import absolute_import, division, print_function
import json
import os
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.urls import basic_auth_header, open_url
from ansible.module_utils._text import to_native
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.utils.display import Display
def grafana_headers(self):
    headers = {'content-type': 'application/json; charset=utf8'}
    if self.grafana_api_key:
        api_key = self.grafana_api_key
        if len(api_key) % 4 == 2:
            display.deprecated('Passing a mangled version of the API key to the grafana_dashboard lookup is no longer necessary and should not be done.', '2.0.0', collection_name='community.grafana')
            api_key += '=='
        headers['Authorization'] = 'Bearer %s' % api_key
    else:
        headers['Authorization'] = basic_auth_header(self.grafana_user, self.grafana_password)
        self.grafana_switch_organisation(headers)
    return headers