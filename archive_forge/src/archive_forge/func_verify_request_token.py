from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def verify_request_token(self, token, request):
    """Verify that the given OAuth1 request token is valid.

        :param token: A request token string.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: True or False

        This method is used only in AuthorizationEndpoint to check whether the
        oauth_token given in the authorization URL is valid or not.
        This request is not signed and thus similar ``validate_request_token``
        method can not be used.

        This method is used by

        * AuthorizationEndpoint
        """
    raise self._subclass_must_implement('verify_request_token')