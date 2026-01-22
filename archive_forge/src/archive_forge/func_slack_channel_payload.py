from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
def slack_channel_payload(data, payload):
    payload['settings']['url'] = data['slack_url']
    if data.get('slack_recipient'):
        payload['settings']['recipient'] = data['slack_recipient']
    if data.get('slack_username'):
        payload['settings']['username'] = data['slack_username']
    if data.get('slack_icon_emoji'):
        payload['settings']['iconEmoji'] = data['slack_icon_emoji']
    if data.get('slack_icon_url'):
        payload['settings']['iconUrl'] = data['slack_icon_url']
    if data.get('slack_mention_users'):
        payload['settings']['mentionUsers'] = ','.join(data['slack_mention_users'])
    if data.get('slack_mention_groups'):
        payload['settings']['mentionGroups'] = ','.join(data['slack_mention_groups'])
    if data.get('slack_mention_channel'):
        payload['settings']['mentionChannel'] = data['slack_mention_channel']
    if data.get('slack_token'):
        payload['settings']['token'] = data['slack_token']