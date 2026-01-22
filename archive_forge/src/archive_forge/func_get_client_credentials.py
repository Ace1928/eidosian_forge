import os
import copy
import hmac
import time
import base64
from hashlib import sha256
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import ET, b, httplib, urlparse, urlencode, basestring
from libcloud.utils.xml import fixxpath
from libcloud.common.base import (
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.common.azure_arm import AzureAuthJsonResponse, publicEnvironments
def get_client_credentials(self):
    """
        Log in and get bearer token used to authorize API requests.
        """
    conn = self.conn_class(self.login_host, 443, timeout=self.timeout)
    conn.connect()
    params = urlencode({'grant_type': 'client_credentials', 'client_id': self.user_id, 'client_secret': self.key, 'resource': 'https://storage.azure.com/'})
    headers = {'Content-type': 'application/x-www-form-urlencoded'}
    conn.request('POST', '/%s/oauth2/token' % self.tenant_id, params, headers)
    js = AzureAuthJsonResponse(conn.getresponse(), conn)
    self.access_token = js.object['access_token']
    self.expires_on = js.object['expires_on']