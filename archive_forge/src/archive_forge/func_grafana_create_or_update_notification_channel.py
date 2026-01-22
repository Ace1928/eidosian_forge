from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def grafana_create_or_update_notification_channel(self, data):
    payload = grafana_notification_channel_payload(data)
    r, info = fetch_url(self._module, '%s/api/alert-notifications/uid/%s' % (data['url'], data['uid']), headers=self.headers)
    if info['status'] == 200:
        before = json.loads(to_text(r.read()))
        return self.grafana_update_notification_channel(data, payload, before)
    elif info['status'] == 404:
        return self.grafana_create_notification_channel(data, payload)
    else:
        raise GrafanaAPIException('Unable to get notification channel %s : %s' % (data['uid'], info))