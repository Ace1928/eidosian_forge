from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate_info import (
def get_certificate_argument_spec():
    return ArgumentSpec(argument_spec=dict(provider=dict(type='str', choices=[]), force=dict(type='bool', default=False), csr_path=dict(type='path'), csr_content=dict(type='str'), ignore_timestamps=dict(type='bool', default=True), select_crypto_backend=dict(type='str', default='auto', choices=['auto', 'cryptography']), privatekey_path=dict(type='path'), privatekey_content=dict(type='str', no_log=True), privatekey_passphrase=dict(type='str', no_log=True)), mutually_exclusive=[['csr_path', 'csr_content'], ['privatekey_path', 'privatekey_content']])