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
def calc_signature(self, request, params):
    logger.debug('Calculating signature using v2 auth.')
    split = urlsplit(request.url)
    path = split.path
    if len(path) == 0:
        path = '/'
    string_to_sign = f'{request.method}\n{split.netloc}\n{path}\n'
    lhmac = hmac.new(self.credentials.secret_key.encode('utf-8'), digestmod=sha256)
    pairs = []
    for key in sorted(params):
        if key == 'Signature':
            continue
        value = str(params[key])
        quoted_key = quote(key.encode('utf-8'), safe='')
        quoted_value = quote(value.encode('utf-8'), safe='-_~')
        pairs.append(f'{quoted_key}={quoted_value}')
    qs = '&'.join(pairs)
    string_to_sign += qs
    logger.debug('String to sign: %s', string_to_sign)
    lhmac.update(string_to_sign.encode('utf-8'))
    b64 = base64.b64encode(lhmac.digest()).strip().decode('utf-8')
    return (qs, b64)