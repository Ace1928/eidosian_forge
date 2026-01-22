from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def hipchat_channel_payload(data, payload):
    payload['settings']['url'] = data['hipchat_url']
    if data.get('hipchat_api_key'):
        payload['settings']['apiKey'] = data['hipchat_api_key']
    if data.get('hipchat_room_id'):
        payload['settings']['roomid'] = data['hipchat_room_id']