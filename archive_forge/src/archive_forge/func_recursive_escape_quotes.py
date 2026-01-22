from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def recursive_escape_quotes(obj, keys):
    """Recursively escape quotes inside supplied keys inside block kit objects"""
    if isinstance(obj, dict):
        escaped = {}
        for k, v in obj.items():
            if isinstance(v, str) and k in keys:
                escaped[k] = escape_quotes(v)
            else:
                escaped[k] = recursive_escape_quotes(v, keys)
    elif isinstance(obj, list):
        escaped = [recursive_escape_quotes(v, keys) for v in obj]
    else:
        escaped = obj
    return escaped