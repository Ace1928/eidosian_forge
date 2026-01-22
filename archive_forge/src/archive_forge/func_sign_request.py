from __future__ import absolute_import, division, print_function
import copy
import datetime
import json
import locale
import time
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import PY3
from ansible_collections.community.crypto.plugins.module_utils.acme.backend_openssl_cli import (
from ansible_collections.community.crypto.plugins.module_utils.acme.backend_cryptography import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
def sign_request(self, protected, payload, key_data, encode_payload=True):
    """
        Signs an ACME request.
        """
    try:
        if payload is None:
            payload64 = ''
        else:
            if encode_payload:
                payload = self.module.jsonify(payload).encode('utf8')
            payload64 = nopad_b64(to_bytes(payload))
        protected64 = nopad_b64(self.module.jsonify(protected).encode('utf8'))
    except Exception as e:
        raise ModuleFailException('Failed to encode payload / headers as JSON: {0}'.format(e))
    return self.backend.sign(payload64, protected64, key_data)