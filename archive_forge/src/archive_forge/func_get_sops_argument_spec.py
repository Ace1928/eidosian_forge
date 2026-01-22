from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text, to_native
from subprocess import Popen, PIPE
def get_sops_argument_spec(add_encrypt_specific=False):
    argument_spec = {'sops_binary': {'type': 'path'}, 'age_key': {'type': 'str', 'no_log': True}, 'age_keyfile': {'type': 'path'}, 'aws_profile': {'type': 'str'}, 'aws_access_key_id': {'type': 'str'}, 'aws_secret_access_key': {'type': 'str', 'no_log': True}, 'aws_session_token': {'type': 'str', 'no_log': True}, 'config_path': {'type': 'path'}, 'enable_local_keyservice': {'type': 'bool', 'default': False}, 'keyservice': {'type': 'list', 'elements': 'str'}}
    if add_encrypt_specific:
        argument_spec.update({'age': {'type': 'list', 'elements': 'str'}, 'kms': {'type': 'list', 'elements': 'str'}, 'gcp_kms': {'type': 'list', 'elements': 'str'}, 'azure_kv': {'type': 'list', 'elements': 'str'}, 'hc_vault_transit': {'type': 'list', 'elements': 'str'}, 'pgp': {'type': 'list', 'elements': 'str'}, 'unencrypted_suffix': {'type': 'str'}, 'encrypted_suffix': {'type': 'str'}, 'unencrypted_regex': {'type': 'str'}, 'encrypted_regex': {'type': 'str'}, 'encryption_context': {'type': 'list', 'elements': 'str'}, 'shamir_secret_sharing_threshold': {'type': 'int', 'no_log': False}})
    return argument_spec