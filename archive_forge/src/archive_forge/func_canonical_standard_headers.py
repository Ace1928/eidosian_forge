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
def canonical_standard_headers(self, headers):
    interesting_headers = ['content-md5', 'content-type', 'date']
    hoi = []
    if 'Date' in headers:
        del headers['Date']
    headers['Date'] = self._get_date()
    for ih in interesting_headers:
        found = False
        for key in headers:
            lk = key.lower()
            if headers[key] is not None and lk == ih:
                hoi.append(headers[key].strip())
                found = True
        if not found:
            hoi.append('')
    return '\n'.join(hoi)