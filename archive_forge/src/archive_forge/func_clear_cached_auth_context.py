import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def clear_cached_auth_context(self):
    """
        Clear the cached authentication context.

        The context is cleared from fields on this connection and from the
        external cache, if one is configured.
        """
    self.auth_token = None
    self.auth_token_expires = None
    self.auth_user_info = None
    self.auth_user_roles = None
    self.urls = {}
    if self.auth_cache is not None:
        self.auth_cache.clear(self._cache_key)