from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def parsed_url_path(self, url):
    if not HAS_URLPARSE:
        self.fail_json(msg='urlparse is not installed')
    parse_result = urlparse(url)
    if parse_result.query == '':
        return parse_result.path
    else:
        return parse_result.path + '?' + parse_result.query