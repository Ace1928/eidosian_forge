import os
import ssl
import sys
from ... import config
from ... import version_string as breezy_version
def cert_reqs_from_store(unicode_str):
    import ssl
    try:
        return {'required': ssl.CERT_REQUIRED, 'none': ssl.CERT_NONE}[unicode_str]
    except KeyError:
        raise ValueError('invalid value %s' % unicode_str)