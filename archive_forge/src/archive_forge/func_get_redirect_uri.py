from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def get_redirect_uri(self, token, request):
    """Get the redirect URI associated with a request token.

        :param token: The request token string.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: The redirect URI associated with the request token.

        It may be desirable to return a custom URI if the redirect is set to
        "oob".
        In this case, the user will be redirected to the returned URI and at
        that
        endpoint the verifier can be displayed.

        This method is used by

        * AuthorizationEndpoint
        """
    raise self._subclass_must_implement('get_redirect_uri')