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
def manage_indicator_led(self, command, resource_uri=None):
    if resource_uri is None:
        resource_uri = self.chassis_uri
    payloads = {'IndicatorLedOn': 'Lit', 'IndicatorLedOff': 'Off', 'IndicatorLedBlink': 'Blinking'}
    if command not in payloads.keys():
        return {'ret': False, 'msg': 'Invalid command (%s)' % command}
    payload = {'IndicatorLED': payloads[command]}
    resp = self.patch_request(self.root_uri + resource_uri, payload, check_pyld=True)
    if resp['ret'] and resp['changed']:
        resp['msg'] = 'Set IndicatorLED to %s' % payloads[command]
    return resp