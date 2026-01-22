import base64
import copy
import datetime
import json
import time
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import crypt
from oauth2client import transport
def create_with_claims(self, claims):
    """Create credentials that specify additional claims.

        Args:
            claims: dict, key-value pairs for claims.

        Returns:
            ServiceAccountCredentials, a copy of the current service account
            credentials with updated claims to use when obtaining access
            tokens.
        """
    new_kwargs = dict(self._kwargs)
    new_kwargs.update(claims)
    result = self.__class__(self._service_account_email, self._signer, scopes=self._scopes, private_key_id=self._private_key_id, client_id=self.client_id, user_agent=self._user_agent, **new_kwargs)
    result.token_uri = self.token_uri
    result.revoke_uri = self.revoke_uri
    result._private_key_pkcs8_pem = self._private_key_pkcs8_pem
    result._private_key_pkcs12 = self._private_key_pkcs12
    result._private_key_password = self._private_key_password
    return result