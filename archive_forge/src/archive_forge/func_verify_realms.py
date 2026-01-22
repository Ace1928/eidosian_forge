from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def verify_realms(self, token, realms, request):
    """Verify authorized realms to see if they match those given to token.

        :param token: An access token string.
        :param realms: A list of realms the client attempts to access.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: True or False

        This prevents the list of authorized realms sent by the client during
        the authorization step to be altered to include realms outside what
        was bound with the request token.

        Can be as simple as::

            valid_realms = self.get_realms(token)
            return all((r in valid_realms for r in realms))

        This method is used by

        * AuthorizationEndpoint
        """
    raise self._subclass_must_implement('verify_realms')