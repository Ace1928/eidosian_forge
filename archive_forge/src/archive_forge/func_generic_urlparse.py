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
def generic_urlparse(parts):
    """
    Returns a dictionary of url parts as parsed by urlparse,
    but accounts for the fact that older versions of that
    library do not support named attributes (ie. .netloc)
    """
    generic_parts = ParseResultDottedDict()
    if hasattr(parts, 'netloc'):
        generic_parts['scheme'] = parts.scheme
        generic_parts['netloc'] = parts.netloc
        generic_parts['path'] = parts.path
        generic_parts['params'] = parts.params
        generic_parts['query'] = parts.query
        generic_parts['fragment'] = parts.fragment
        generic_parts['username'] = parts.username
        generic_parts['password'] = parts.password
        hostname = parts.hostname
        if hostname and hostname[0] == '[' and ('[' in parts.netloc) and (']' in parts.netloc):
            hostname = parts.netloc.split(']')[0][1:].lower()
        generic_parts['hostname'] = hostname
        try:
            port = parts.port
        except ValueError:
            netloc = parts.netloc.split('@')[-1].split(']')[-1]
            if ':' in netloc:
                port = netloc.split(':')[1]
                if port:
                    port = int(port)
            else:
                port = None
        generic_parts['port'] = port
    else:
        generic_parts['scheme'] = parts[0]
        generic_parts['netloc'] = parts[1]
        generic_parts['path'] = parts[2]
        generic_parts['params'] = parts[3]
        generic_parts['query'] = parts[4]
        generic_parts['fragment'] = parts[5]
        try:
            netloc_re = re.compile('^((?:\\w)+(?::(?:\\w)+)?@)?([A-Za-z0-9.-]+)(:\\d+)?$')
            match = netloc_re.match(parts[1])
            auth = match.group(1)
            hostname = match.group(2)
            port = match.group(3)
            if port:
                port = int(port[1:])
            if auth:
                auth = auth[:-1]
                username, password = auth.split(':', 1)
            else:
                username = password = None
            generic_parts['username'] = username
            generic_parts['password'] = password
            generic_parts['hostname'] = hostname
            generic_parts['port'] = port
        except Exception:
            generic_parts['username'] = None
            generic_parts['password'] = None
            generic_parts['hostname'] = parts[1]
            generic_parts['port'] = None
    return generic_parts