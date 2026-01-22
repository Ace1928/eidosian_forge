from boto.connection import AWSQueryConnection
from boto.provider import Provider, NO_CREDENTIALS_PROVIDED
from boto.regioninfo import RegionInfo
from boto.sts.credentials import Credentials, FederationToken, AssumedRole
from boto.sts.credentials import DecodeAuthorizationMessage
import boto
import boto.utils
import datetime
import threading
def get_session_token(self, duration=None, force_new=False, mfa_serial_number=None, mfa_token=None):
    """
        Return a valid session token.  Because retrieving new tokens
        from the Secure Token Service is a fairly heavyweight operation
        this module caches previously retrieved tokens and returns
        them when appropriate.  Each token is cached with a key
        consisting of the region name of the STS endpoint
        concatenated with the requesting user's access id.  If there
        is a token in the cache meeting with this key, the session
        expiration is checked to make sure it is still valid and if
        so, the cached token is returned.  Otherwise, a new session
        token is requested from STS and it is placed into the cache
        and returned.

        :type duration: int
        :param duration: The number of seconds the credentials should
            remain valid.

        :type force_new: bool
        :param force_new: If this parameter is True, a new session token
            will be retrieved from the Secure Token Service regardless
            of whether there is a valid cached token or not.

        :type mfa_serial_number: str
        :param mfa_serial_number: The serial number of an MFA device.
            If this is provided and if the mfa_passcode provided is
            valid, the temporary session token will be authorized with
            to perform operations requiring the MFA device authentication.

        :type mfa_token: str
        :param mfa_token: The 6 digit token associated with the
            MFA device.
        """
    token_key = '%s:%s' % (self.region.name, self.provider.access_key)
    token = self._check_token_cache(token_key, duration)
    if force_new or not token:
        boto.log.debug('fetching a new token for %s' % token_key)
        try:
            self._mutex.acquire()
            token = self._get_session_token(duration, mfa_serial_number, mfa_token)
            _session_token_cache[token_key] = token
        finally:
            self._mutex.release()
    return token