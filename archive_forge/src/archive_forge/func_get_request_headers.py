import hmac
import time
import base64
import hashlib
from typing import Dict, Type, Optional
from hashlib import sha256
from datetime import datetime
from libcloud.utils.py3 import ET, b, httplib, urlquote, basestring, _real_unicode
from libcloud.utils.xml import findall_ignore_namespace, findtext_ignore_namespace
from libcloud.common.base import BaseDriver, XmlResponse, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
def get_request_headers(self, params, headers, method='GET', path='/', data=None):
    now = datetime.utcnow()
    headers['X-AMZ-Date'] = now.strftime('%Y%m%dT%H%M%SZ')
    headers['X-AMZ-Content-SHA256'] = self._get_payload_hash(method, data)
    headers['Authorization'] = self._get_authorization_v4_header(params=params, headers=headers, dt=now, method=method, path=path, data=data)
    return (params, headers)