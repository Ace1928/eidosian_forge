from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def pagerduty_channel_payload(data, payload):
    payload['settings']['integrationKey'] = data['pagerduty_integration_key']
    if data.get('pagerduty_severity'):
        payload['settings']['severity'] = data['pagerduty_severity']
    if data.get('pagerduty_auto_resolve'):
        payload['settings']['autoResolve'] = data['pagerduty_auto_resolve']
    if data.get('pagerduty_message_in_details'):
        payload['settings']['messageInDetails'] = data['pagerduty_message_in_details']