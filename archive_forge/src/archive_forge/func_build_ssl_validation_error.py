from __future__ import (absolute_import, division, print_function)
import atexit
import base64
import email.mime.multipart
import email.mime.nonmultipart
import email.mime.application
import email.parser
import email.utils
import functools
import io
import mimetypes
import netrc
import os
import platform
import re
import socket
import sys
import tempfile
import traceback
import types  # pylint: disable=unused-import
from contextlib import contextmanager
import ansible.module_utils.compat.typing as t
import ansible.module_utils.six.moves.http_cookiejar as cookiejar
import ansible.module_utils.six.moves.urllib.error as urllib_error
from ansible.module_utils.common.collections import Mapping, is_sequence
from ansible.module_utils.six import PY2, PY3, string_types
from ansible.module_utils.six.moves import cStringIO
from ansible.module_utils.basic import get_distribution, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def build_ssl_validation_error(hostname, port, paths, exc=None):
    """Inteligently build out the SSLValidationError based on what support
    you have installed
    """
    msg = ['Failed to validate the SSL certificate for %s:%s. Make sure your managed systems have a valid CA certificate installed.']
    if not HAS_SSLCONTEXT:
        msg.append('If the website serving the url uses SNI you need python >= 2.7.9 on your managed machine')
        msg.append(' (the python executable used (%s) is version: %s)' % (sys.executable, ''.join(sys.version.splitlines())))
        if not HAS_URLLIB3_PYOPENSSLCONTEXT and (not HAS_URLLIB3_SSL_WRAP_SOCKET):
            msg.append('or you can install the `urllib3`, `pyOpenSSL`, `ndg-httpsclient`, and `pyasn1` python modules')
        msg.append('to perform SNI verification in python >= 2.6.')
    msg.append('You can use validate_certs=False if you do not need to confirm the servers identity but this is unsafe and not recommended. Paths checked for this platform: %s.')
    if exc:
        msg.append('The exception msg was: %s.' % to_native(exc))
    raise SSLValidationError(' '.join(msg) % (hostname, port, ', '.join(paths)))