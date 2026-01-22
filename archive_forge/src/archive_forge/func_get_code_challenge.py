from __future__ import absolute_import, unicode_literals
import logging
def get_code_challenge(self, code, request):
    """Is called for every "token" requests.

        When the server issues the authorization code in the authorization
        response, it MUST associate the ``code_challenge`` and
        ``code_challenge_method`` values with the authorization code so it can
        be verified later.

        Typically, the ``code_challenge`` and ``code_challenge_method`` values
        are stored in encrypted form in the ``code`` itself but could
        alternatively be stored on the server associated with the code.  The
        server MUST NOT include the ``code_challenge`` value in client requests
        in a form that other entities can extract.

        Return the ``code_challenge`` associated to the code.
        If ``None`` is returned, code is considered to not be associated to any
        challenges.

        :param code: Authorization code.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: code_challenge string

        Method is used by:
            - Authorization Code Grant - when PKCE is active

        """
    return None