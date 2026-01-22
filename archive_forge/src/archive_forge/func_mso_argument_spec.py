from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def mso_argument_spec():
    return dict(host=dict(type='str', required=False, aliases=['hostname'], fallback=(env_fallback, ['MSO_HOST'])), port=dict(type='int', required=False, fallback=(env_fallback, ['MSO_PORT'])), username=dict(type='str', required=False, fallback=(env_fallback, ['MSO_USERNAME', 'ANSIBLE_NET_USERNAME'])), password=dict(type='str', required=False, no_log=True, fallback=(env_fallback, ['MSO_PASSWORD', 'ANSIBLE_NET_PASSWORD'])), output_level=dict(type='str', default='normal', choices=['debug', 'info', 'normal'], fallback=(env_fallback, ['MSO_OUTPUT_LEVEL'])), timeout=dict(type='int', fallback=(env_fallback, ['MSO_TIMEOUT'])), use_proxy=dict(type='bool', fallback=(env_fallback, ['MSO_USE_PROXY'])), use_ssl=dict(type='bool', fallback=(env_fallback, ['MSO_USE_SSL'])), validate_certs=dict(type='bool', fallback=(env_fallback, ['MSO_VALIDATE_CERTS'])), login_domain=dict(type='str', fallback=(env_fallback, ['MSO_LOGIN_DOMAIN'])))