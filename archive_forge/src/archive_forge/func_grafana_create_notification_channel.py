from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def grafana_create_notification_channel(self, data, payload):
    r, info = fetch_url(self._module, '%s/api/alert-notifications' % data['url'], data=json.dumps(payload), headers=self.headers, method='POST')
    if info['status'] == 200:
        return {'state': 'present', 'changed': True, 'channel': json.loads(to_text(r.read()))}
    else:
        raise GrafanaAPIException('Unable to create notification channel: %s' % info)