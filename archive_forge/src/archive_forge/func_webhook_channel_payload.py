from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def webhook_channel_payload(data, payload):
    payload['settings']['url'] = data['webhook_url']
    if data.get('webhook_http_method'):
        payload['settings']['httpMethod'] = data['webhook_http_method']
    if data.get('webhook_username'):
        payload['settings']['username'] = data['webhook_username']
    if data.get('webhook_password'):
        payload['settings']['password'] = data['webhook_password']