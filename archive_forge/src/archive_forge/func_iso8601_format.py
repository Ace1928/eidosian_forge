from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def iso8601_format(self, dt):
    """Return an ACI-compatible ISO8601 formatted time: 2123-12-12T00:00:00.000+00:00"""
    try:
        return dt.isoformat(timespec='milliseconds')
    except Exception:
        tz = dt.strftime('%z')
        return '%s.%03d%s:%s' % (dt.strftime('%Y-%m-%dT%H:%M:%S'), dt.microsecond / 1000, tz[:3], tz[3:])