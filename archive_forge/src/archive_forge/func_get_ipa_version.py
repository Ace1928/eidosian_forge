from __future__ import (absolute_import, division, print_function)
import json
import os
import socket
import uuid
import re
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url, HAS_GSSAPI
from ansible.module_utils.basic import env_fallback, AnsibleFallbackNotFound
def get_ipa_version(self):
    response = self.ping()['summary']
    ipa_ver_regex = re.compile('IPA server version (\\d\\.\\d\\.\\d).*')
    version_match = ipa_ver_regex.match(response)
    ipa_version = None
    if version_match:
        ipa_version = version_match.groups()[0]
    return ipa_version