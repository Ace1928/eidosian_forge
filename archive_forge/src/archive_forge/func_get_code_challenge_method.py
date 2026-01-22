from __future__ import absolute_import, unicode_literals
import logging
def get_code_challenge_method(self, code, request):
    """Is called during the "token" request processing, when a

        ``code_verifier`` and a ``code_challenge`` has been provided.

        See ``.get_code_challenge``.

        Must return ``plain`` or ``S256``. You can return a custom value if you
        have
        implemented your own ``AuthorizationCodeGrant`` class.

        :param code: Authorization code.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: code_challenge_method string

        Method is used by:
            - Authorization Code Grant - when PKCE is active

        """
    raise NotImplementedError('Subclasses must implement this method.')