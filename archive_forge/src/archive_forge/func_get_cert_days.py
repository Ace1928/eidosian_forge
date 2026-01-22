from __future__ import absolute_import, division, print_function
import base64
import binascii
import datetime
import os
import re
import tempfile
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.acme.backends import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import nopad_b64
def get_cert_days(self, cert_filename=None, cert_content=None, now=None):
    """
        Return the days the certificate in cert_filename remains valid and -1
        if the file was not found. If cert_filename contains more than one
        certificate, only the first one will be considered.

        If now is not specified, datetime.datetime.now() is used.
        """
    filename = cert_filename
    data = None
    if cert_content is not None:
        filename = '/dev/stdin'
        data = cert_content.encode('utf-8')
        cert_filename_suffix = ''
    elif cert_filename is not None:
        if not os.path.exists(cert_filename):
            return -1
        cert_filename_suffix = ' in {0}'.format(cert_filename)
    else:
        return -1
    openssl_cert_cmd = [self.openssl_binary, 'x509', '-in', filename, '-noout', '-text']
    dummy, out, dummy = self.module.run_command(openssl_cert_cmd, data=data, check_rc=True, binary_data=True, environ_update=_OPENSSL_ENVIRONMENT_UPDATE)
    try:
        not_after_str = re.search('\\s+Not After\\s*:\\s+(.*)', to_text(out, errors='surrogate_or_strict')).group(1)
        not_after = datetime.datetime.strptime(not_after_str, '%b %d %H:%M:%S %Y %Z')
    except AttributeError:
        raise BackendException("No 'Not after' date found{0}".format(cert_filename_suffix))
    except ValueError:
        raise BackendException("Failed to parse 'Not after' date{0}".format(cert_filename_suffix))
    if now is None:
        now = datetime.datetime.now()
    return (not_after - now).days