from __future__ import absolute_import, division, print_function
import copy
import json
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
def camel_case_key(key):
    parts = []
    for part in key.split('_'):
        if part in {'id', 'ttl', 'jwks', 'jwt', 'oidc', 'iam', 'sts'}:
            parts.append(part.upper())
        else:
            parts.append(part.capitalize())
    return ''.join(parts)