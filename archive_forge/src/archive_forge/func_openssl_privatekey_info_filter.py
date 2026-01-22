from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.privatekey_info import (
from ansible_collections.community.crypto.plugins.plugin_utils.filter_module import FilterModuleMock
def openssl_privatekey_info_filter(data, passphrase=None, return_private_key_data=False):
    """Extract information from X.509 PEM certificate."""
    if not isinstance(data, string_types):
        raise AnsibleFilterError('The community.crypto.openssl_privatekey_info input must be a text type, not %s' % type(data))
    if passphrase is not None and (not isinstance(passphrase, string_types)):
        raise AnsibleFilterError('The passphrase option must be a text type, not %s' % type(passphrase))
    if not isinstance(return_private_key_data, bool):
        raise AnsibleFilterError('The return_private_key_data option must be a boolean, not %s' % type(return_private_key_data))
    module = FilterModuleMock({})
    try:
        result = get_privatekey_info(module, 'cryptography', content=to_bytes(data), passphrase=passphrase, return_private_key_data=return_private_key_data)
        result.pop('can_parse_key', None)
        result.pop('key_is_consistent', None)
        return result
    except PrivateKeyParseError as exc:
        raise AnsibleFilterError(exc.error_message)
    except OpenSSLObjectError as exc:
        raise AnsibleFilterError(to_native(exc))