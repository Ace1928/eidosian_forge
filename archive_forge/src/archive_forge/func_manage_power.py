from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def manage_power(self, command, resource_uri, action_name):
    key = 'Actions'
    reset_type_values = ['On', 'ForceOff', 'GracefulShutdown', 'GracefulRestart', 'ForceRestart', 'Nmi', 'ForceOn', 'PushPowerButton', 'PowerCycle']
    if not command.startswith('Power'):
        return {'ret': False, 'msg': 'Invalid Command (%s)' % command}
    if command == 'PowerCycle':
        reset_type = command
    else:
        reset_type = command[5:]
    if reset_type == 'Reboot':
        reset_type = 'GracefulRestart'
    if reset_type not in reset_type_values:
        return {'ret': False, 'msg': 'Invalid Command (%s)' % command}
    response = self.get_request(self.root_uri + resource_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    power_state = data.get('PowerState')
    if power_state == 'On' and reset_type in ['On', 'ForceOn']:
        return {'ret': True, 'changed': False}
    if power_state == 'Off' and reset_type in ['GracefulShutdown', 'ForceOff']:
        return {'ret': True, 'changed': False}
    if key not in data or action_name not in data[key]:
        return {'ret': False, 'msg': 'Action %s not found' % action_name}
    reset_action = data[key][action_name]
    if 'target' not in reset_action:
        return {'ret': False, 'msg': 'target URI missing from Action %s' % action_name}
    action_uri = reset_action['target']
    ai = self._get_all_action_info_values(reset_action)
    allowable_values = ai.get('ResetType', {}).get('AllowableValues', [])
    if reset_type not in allowable_values:
        reset_type = self._map_reset_type(reset_type, allowable_values)
    payload = {'ResetType': reset_type}
    response = self.post_request(self.root_uri + action_uri, payload)
    if response['ret'] is False:
        return response
    return {'ret': True, 'changed': True}