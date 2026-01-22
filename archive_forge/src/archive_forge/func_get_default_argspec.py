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
def get_default_argspec():
    """
    Provides default argument spec for the options documented in the acme doc fragment.
    """
    return dict(account_key_src=dict(type='path', aliases=['account_key']), account_key_content=dict(type='str', no_log=True), account_key_passphrase=dict(type='str', no_log=True), account_uri=dict(type='str'), acme_directory=dict(type='str', required=True), acme_version=dict(type='int', required=True, choices=[1, 2]), validate_certs=dict(type='bool', default=True), select_crypto_backend=dict(type='str', default='auto', choices=['auto', 'openssl', 'cryptography']), request_timeout=dict(type='int', default=10))