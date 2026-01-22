from __future__ import absolute_import, division, print_function
from .basic import (
from .cryptography_support import (
from ._obj2txt import (
def cryptography_decode_revoked_certificate(cert):
    result = {'serial_number': cert.serial_number, 'revocation_date': cert.revocation_date, 'issuer': None, 'issuer_critical': False, 'reason': None, 'reason_critical': False, 'invalidity_date': None, 'invalidity_date_critical': False}
    try:
        ext = cert.extensions.get_extension_for_class(x509.CertificateIssuer)
        result['issuer'] = list(ext.value)
        result['issuer_critical'] = ext.critical
    except x509.ExtensionNotFound:
        pass
    try:
        ext = cert.extensions.get_extension_for_class(x509.CRLReason)
        result['reason'] = ext.value.reason
        result['reason_critical'] = ext.critical
    except x509.ExtensionNotFound:
        pass
    try:
        ext = cert.extensions.get_extension_for_class(x509.InvalidityDate)
        result['invalidity_date'] = ext.value.invalidity_date
        result['invalidity_date_critical'] = ext.critical
    except x509.ExtensionNotFound:
        pass
    return result