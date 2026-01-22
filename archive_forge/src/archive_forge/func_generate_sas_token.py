from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def generate_sas_token(self, **kwags):
    base_url = kwags.get('base_url', None)
    expiry = kwags.get('expiry', time() + 3600)
    key = kwags.get('key', None)
    policy = kwags.get('policy', None)
    url = quote_plus(base_url)
    ttl = int(expiry)
    sign_key = '{0}\n{1}'.format(url, ttl)
    signature = b64encode(HMAC(b64decode(key), sign_key.encode('utf-8'), sha256).digest())
    result = {'sr': url, 'sig': signature, 'se': str(ttl)}
    if policy:
        result['skn'] = policy
    return 'SharedAccessSignature ' + urlencode(result)