from __future__ import absolute_import, division, print_function
import base64
import binascii
import re
import sys
import traceback
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse, ParseResult
from ._asn1 import serialize_asn1_string_as_der
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import missing_required_lib
from .basic import (
from ._objects import (
from ._obj2txt import obj2txt
def cryptography_get_name(name, what='Subject Alternative Name'):
    """
    Given a name string, returns a cryptography x509.GeneralName object.
    Raises an OpenSSLObjectError if the name is unknown or cannot be parsed.
    """
    try:
        if name.startswith('DNS:'):
            return x509.DNSName(_adjust_idn(to_text(name[4:]), 'idna'))
        if name.startswith('IP:'):
            address = to_text(name[3:])
            if '/' in address:
                return x509.IPAddress(ipaddress.ip_network(address))
            return x509.IPAddress(ipaddress.ip_address(address))
        if name.startswith('email:'):
            return x509.RFC822Name(_adjust_idn_email(to_text(name[6:]), 'idna'))
        if name.startswith('URI:'):
            return x509.UniformResourceIdentifier(_adjust_idn_url(to_text(name[4:]), 'idna'))
        if name.startswith('RID:'):
            m = re.match('^([0-9]+(?:\\.[0-9]+)*)$', to_text(name[4:]))
            if not m:
                raise OpenSSLObjectError('Cannot parse {what} "{name}"'.format(name=name, what=what))
            return x509.RegisteredID(x509.oid.ObjectIdentifier(m.group(1)))
        if name.startswith('otherName:'):
            m = re.match('^([0-9]+(?:\\.[0-9]+)*);([0-9a-fA-F]{1,2}(?::[0-9a-fA-F]{1,2})*)$', to_text(name[10:]))
            if m:
                return x509.OtherName(x509.oid.ObjectIdentifier(m.group(1)), _parse_hex(m.group(2)))
            name = to_text(name[10:], errors='surrogate_or_strict')
            if ';' not in name:
                raise OpenSSLObjectError('Cannot parse {what} otherName "{name}", must be in the format "otherName:<OID>;<ASN.1 OpenSSL Encoded String>" or "otherName:<OID>;<hex string>"'.format(name=name, what=what))
            oid, value = name.split(';', 1)
            b_value = serialize_asn1_string_as_der(value)
            return x509.OtherName(x509.ObjectIdentifier(oid), b_value)
        if name.startswith('dirName:'):
            return x509.DirectoryName(x509.Name(reversed(_parse_dn(to_bytes(name[8:])))))
    except Exception as e:
        raise OpenSSLObjectError('Cannot parse {what} "{name}": {error}'.format(name=name, what=what, error=e))
    if ':' not in name:
        raise OpenSSLObjectError('Cannot parse {what} "{name}" (forgot "DNS:" prefix?)'.format(name=name, what=what))
    raise OpenSSLObjectError('Cannot parse {what} "{name}" (potentially unsupported by cryptography backend)'.format(name=name, what=what))