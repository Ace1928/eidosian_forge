from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def validate_realms(self, client_key, token, request, uri=None, realms=None):
    """Validates access to the request realm.

        :param client_key: The client/consumer key.
        :param token: A request token string.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :param uri: The URI the realms is protecting.
        :param realms: A list of realms that must have been granted to
                       the access token.
        :returns: True or False

        How providers choose to use the realm parameter is outside the OAuth
        specification but it is commonly used to restrict access to a subset
        of protected resources such as "photos".

        realms is a convenience parameter which can be used to provide
        a per view method pre-defined list of allowed realms.

        Can be as simple as::

            from your_datastore import RequestToken
            request_token = RequestToken.get(token, None)

            if not request_token:
                return False
            return set(request_token.realms).issuperset(set(realms))

        This method is used by

        * ResourceEndpoint
        """
    raise self._subclass_must_implement('validate_realms')