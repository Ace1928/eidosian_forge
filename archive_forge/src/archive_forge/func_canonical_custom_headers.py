import base64
import calendar
import datetime
import functools
import hmac
import json
import logging
import time
from collections.abc import Mapping
from email.utils import formatdate
from hashlib import sha1, sha256
from operator import itemgetter
from botocore.compat import (
from botocore.exceptions import NoAuthTokenError, NoCredentialsError
from botocore.utils import (
from botocore.compat import MD5_AVAILABLE  # noqa
def canonical_custom_headers(self, headers):
    hoi = []
    custom_headers = {}
    for key in headers:
        lk = key.lower()
        if headers[key] is not None:
            if lk.startswith('x-amz-'):
                custom_headers[lk] = ','.join((v.strip() for v in headers.get_all(key)))
    sorted_header_keys = sorted(custom_headers.keys())
    for key in sorted_header_keys:
        hoi.append(f'{key}:{custom_headers[key]}')
    return '\n'.join(hoi)