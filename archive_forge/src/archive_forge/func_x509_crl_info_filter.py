from __future__ import absolute_import, division, print_function
import base64
import binascii
from ansible.errors import AnsibleFilterError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.crl_info import (
from ansible_collections.community.crypto.plugins.plugin_utils.filter_module import FilterModuleMock
def x509_crl_info_filter(data, name_encoding='ignore', list_revoked_certificates=True):
    """Extract information from X.509 PEM certificate."""
    if not isinstance(data, string_types):
        raise AnsibleFilterError('The community.crypto.x509_crl_info input must be a text type, not %s' % type(data))
    if not isinstance(name_encoding, string_types):
        raise AnsibleFilterError('The name_encoding option must be of a text type, not %s' % type(name_encoding))
    if not isinstance(list_revoked_certificates, bool):
        raise AnsibleFilterError('The list_revoked_certificates option must be a boolean, not %s' % type(list_revoked_certificates))
    name_encoding = to_native(name_encoding)
    if name_encoding not in ('ignore', 'idna', 'unicode'):
        raise AnsibleFilterError('The name_encoding option must be one of the values "ignore", "idna", or "unicode", not "%s"' % name_encoding)
    data = to_bytes(data)
    if not identify_pem_format(data):
        try:
            data = base64.b64decode(to_native(data))
        except (binascii.Error, TypeError, ValueError, UnicodeEncodeError) as e:
            pass
    module = FilterModuleMock({'name_encoding': name_encoding})
    try:
        return get_crl_info(module, content=data, list_revoked_certificates=list_revoked_certificates)
    except OpenSSLObjectError as exc:
        raise AnsibleFilterError(to_native(exc))