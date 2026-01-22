from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest
import json
def decode_response(response, module):
    if 'name' in response:
        response['name'] = response['name'].split('/')[-1]
    return response