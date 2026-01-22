from __future__ import (absolute_import, division, print_function)
from base64 import b64encode
from email.utils import formatdate
import re
import json
import hashlib
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
def configure_resource(self, moid, resource_path, body, query_params, update_method=''):
    if not self.module.check_mode:
        if moid and update_method != 'post':
            options = {'http_method': 'patch', 'resource_path': resource_path, 'body': body, 'moid': moid}
            response_dict = self.call_api(**options)
            if response_dict.get('Results'):
                self.result['api_response'] = response_dict['Results'][0]
                self.result['trace_id'] = response_dict.get('trace_id')
        else:
            options = {'http_method': 'post', 'resource_path': resource_path, 'body': body}
            response_dict = self.call_api(**options)
            if response_dict:
                self.result['api_response'] = response_dict
                self.result['trace_id'] = response_dict.get('trace_id')
            elif query_params:
                self.get_resource(resource_path=resource_path, query_params=query_params)
    self.result['changed'] = True