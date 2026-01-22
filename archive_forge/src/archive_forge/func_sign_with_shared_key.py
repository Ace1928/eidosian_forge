import base64
import binascii
import calendar
import concurrent.futures
import datetime
import hashlib
import hmac
import json
import math
import os
import re
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import urllib3
from blobfile import _common as common
from blobfile import _xml as xml
from blobfile._common import (
def sign_with_shared_key(req: Request, key: str) -> str:
    params_to_sign = []
    if req.params is not None:
        for name, value in req.params.items():
            canonical_name = name.lower()
            params_to_sign.append(f'{canonical_name}:{value}')
    u = urllib.parse.urlparse(req.url)
    storage_account = u.netloc.split('.')[0]
    canonical_url = f'/{storage_account}/{u.path[1:]}'
    canonicalized_resource = '\n'.join([canonical_url] + list(sorted(params_to_sign)))
    if req.headers is None:
        headers = {}
    else:
        headers = dict(req.headers)
    headers_to_sign = []
    for name, value in headers.items():
        canonical_name = name.lower()
        canonical_value = re.sub('\\s+', ' ', value).strip()
        if canonical_name.startswith('x-ms-'):
            headers_to_sign.append(f'{canonical_name}:{canonical_value}')
    canonicalized_headers = '\n'.join(sorted(headers_to_sign))
    content_length = headers.get('Content-Length', '')
    if req.data is not None:
        content_length = str(len(req.data))
    parts_to_sign = [req.method, headers.get('Content-Encoding', ''), headers.get('Content-Language', ''), content_length, headers.get('Content-MD5', ''), headers.get('Content-Type', ''), headers.get('Date', ''), headers.get('If-Modified-Since', ''), headers.get('If-Match', ''), headers.get('If-None-Match', ''), headers.get('If-Unmodified-Since', ''), headers.get('Range', ''), canonicalized_headers, canonicalized_resource]
    string_to_sign = '\n'.join(parts_to_sign)
    signature = base64.b64encode(hmac.digest(base64.b64decode(key), string_to_sign.encode('utf8'), 'sha256')).decode('utf8')
    return f'SharedKey {storage_account}:{signature}'