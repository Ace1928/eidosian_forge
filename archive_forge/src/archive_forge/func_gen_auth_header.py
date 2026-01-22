import hmac
import json
import base64
import datetime
from hashlib import sha256
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
def gen_auth_header(self, api_key, secret_key, method, url, timestamp):
    signature = self.calculate_auth_signature(secret_key, method, url, timestamp)
    auth_b64 = base64.b64encode(b('{}:{}'.format(api_key, signature)))
    return 'AuroraDNSv1 %s' % auth_b64.decode('utf-8')