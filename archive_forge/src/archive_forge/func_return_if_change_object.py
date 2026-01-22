from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import copy
import datetime
import time
def return_if_change_object(module, response):
    if response.status_code == 404:
        return None
    if response.status_code == 204:
        return None
    try:
        response.raise_for_status()
        result = response.json()
    except getattr(json.decoder, 'JSONDecodeError', ValueError) as inst:
        module.fail_json(msg='Invalid JSON response with error: %s' % inst)
    if result['kind'] != 'dns#change':
        module.fail_json(msg='Invalid result: %s' % result['kind'])
    return result