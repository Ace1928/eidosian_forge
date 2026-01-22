from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def get_client_secret(self, client_key, request):
    """Retrieves the client secret associated with the client key.

        :param client_key: The client/consumer key.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: The client secret as a string.

        This method must allow the use of a dummy client_key value.
        Fetching the secret using the dummy key must take the same amount of
        time as fetching a secret for a valid client::

            # Unlikely to be near constant time as it uses two database
            # lookups for a valid client, and only one for an invalid.
            from your_datastore import ClientSecret
            if ClientSecret.has(client_key):
                return ClientSecret.get(client_key)
            else:
                return 'dummy'

            # Aim to mimic number of latency inducing operations no matter
            # whether the client is valid or not.
            from your_datastore import ClientSecret
            return ClientSecret.get(client_key, 'dummy')

        Note that the returned key must be in plaintext.

        This method is used by

        * AccessTokenEndpoint
        * RequestTokenEndpoint
        * ResourceEndpoint
        * SignatureOnlyEndpoint
        """
    raise self._subclass_must_implement('get_client_secret')