import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def is_token_valid(self):
    """
        Return True if the current auth token is already cached and hasn't
        expired yet.

        :return: ``True`` if the token is still valid, ``False`` otherwise.
        :rtype: ``bool``
        """
    if not self.auth_token:
        return False
    if not self.auth_token_expires:
        return False
    expires = self.auth_token_expires - datetime.timedelta(seconds=AUTH_TOKEN_EXPIRES_GRACE_SECONDS)
    time_tuple_expires = expires.utctimetuple()
    time_tuple_now = datetime.datetime.utcnow().utctimetuple()
    if time_tuple_now < time_tuple_expires:
        return True
    return False