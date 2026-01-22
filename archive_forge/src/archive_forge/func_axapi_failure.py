from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import fetch_url
def axapi_failure(result):
    if 'response' in result and result['response'].get('status') == 'fail':
        return True
    return False