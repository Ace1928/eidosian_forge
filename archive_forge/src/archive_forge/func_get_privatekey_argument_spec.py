from __future__ import absolute_import, division, print_function
import abc
import base64
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.privatekey_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def get_privatekey_argument_spec():
    return ArgumentSpec(argument_spec=dict(size=dict(type='int', default=4096), type=dict(type='str', default='RSA', choices=['DSA', 'ECC', 'Ed25519', 'Ed448', 'RSA', 'X25519', 'X448']), curve=dict(type='str', choices=['secp224r1', 'secp256k1', 'secp256r1', 'secp384r1', 'secp521r1', 'secp192r1', 'brainpoolP256r1', 'brainpoolP384r1', 'brainpoolP512r1', 'sect163k1', 'sect163r2', 'sect233k1', 'sect233r1', 'sect283k1', 'sect283r1', 'sect409k1', 'sect409r1', 'sect571k1', 'sect571r1']), passphrase=dict(type='str', no_log=True), cipher=dict(type='str'), format=dict(type='str', default='auto_ignore', choices=['pkcs1', 'pkcs8', 'raw', 'auto', 'auto_ignore']), format_mismatch=dict(type='str', default='regenerate', choices=['regenerate', 'convert']), select_crypto_backend=dict(type='str', choices=['auto', 'cryptography'], default='auto'), regenerate=dict(type='str', default='full_idempotence', choices=['never', 'fail', 'partial_idempotence', 'full_idempotence', 'always'])), required_together=[['cipher', 'passphrase']], required_if=[['type', 'ECC', ['curve']]])