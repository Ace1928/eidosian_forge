from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def validate_verifier(self, client_key, token, verifier, request):
    """Validates a verification code.

        :param client_key: The client/consumer key.
        :param token: A request token string.
        :param verifier: The authorization verifier string.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: True or False

        OAuth providers issue a verification code to clients after the
        resource owner authorizes access. This code is used by the client to
        obtain token credentials and the provider must verify that the
        verifier is valid and associated with the client as well as the
        resource owner.

        Verifier validation should be done in near constant time
        (to avoid verifier enumeration). To achieve this we need a
        constant time string comparison which is provided by OAuthLib
        in ``oauthlib.common.safe_string_equals``::

            from your_datastore import Verifier
            correct_verifier = Verifier.get(client_key, request_token)
            from oauthlib.common import safe_string_equals
            return safe_string_equals(verifier, correct_verifier)

        This method is used by

        * AccessTokenEndpoint
        """
    raise self._subclass_must_implement('validate_verifier')