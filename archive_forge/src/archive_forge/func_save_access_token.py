from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def save_access_token(self, token, request):
    """Save an OAuth1 access token.

        :param token: A dict with token credentials.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request

        The token dictionary will at minimum include

        * ``oauth_token`` the access token string.
        * ``oauth_token_secret`` the token specific secret used in signing.
        * ``oauth_authorized_realms`` a space separated list of realms.

        Client key can be obtained from ``request.client_key``.

        The list of realms (not joined string) can be obtained from
        ``request.realm``.

        This method is used by

        * AccessTokenEndpoint
        """
    raise self._subclass_must_implement('save_access_token')