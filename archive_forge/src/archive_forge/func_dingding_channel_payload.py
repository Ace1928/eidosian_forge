from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def dingding_channel_payload(data, payload):
    payload['settings']['url'] = data['dingding_url']
    if data.get('dingding_message_type'):
        payload['settings']['msgType'] = {'link': 'link', 'action_card': 'actionCard'}[data['dingding_message_type']]