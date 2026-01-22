from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def discord_channel_payload(data, payload):
    payload['settings']['url'] = data['discord_url']
    if data.get('discord_message_content'):
        payload['settings']['content'] = data['discord_message_content']