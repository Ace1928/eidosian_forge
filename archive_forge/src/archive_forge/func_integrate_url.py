from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def integrate_url(httpapi_url, local_path):
    parse_url = urlparse(httpapi_url)
    return {'protocol': parse_url.scheme, 'host': parse_url.netloc, 'path': local_path}