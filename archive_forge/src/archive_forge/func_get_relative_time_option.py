from __future__ import absolute_import, division, print_function
import abc
import datetime
import errno
import hashlib
import os
import re
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from .basic import (
def get_relative_time_option(input_string, input_name, backend='cryptography'):
    """Return an absolute timespec if a relative timespec or an ASN1 formatted
       string is provided.

       The return value will be a datetime object for the cryptography backend,
       and a ASN1 formatted string for the pyopenssl backend."""
    result = to_native(input_string)
    if result is None:
        raise OpenSSLObjectError('The timespec "%s" for %s is not valid' % input_string, input_name)
    if result.startswith('+') or result.startswith('-'):
        result_datetime = convert_relative_to_datetime(result)
        if backend == 'pyopenssl':
            return result_datetime.strftime('%Y%m%d%H%M%SZ')
        elif backend == 'cryptography':
            return result_datetime
    if backend == 'cryptography':
        for date_fmt in ['%Y%m%d%H%M%SZ', '%Y%m%d%H%MZ', '%Y%m%d%H%M%S%z', '%Y%m%d%H%M%z']:
            try:
                return datetime.datetime.strptime(result, date_fmt)
            except ValueError:
                pass
        raise OpenSSLObjectError('The time spec "%s" for %s is invalid' % (input_string, input_name))