from __future__ import absolute_import, unicode_literals
import logging
def validate_user_match(self, id_token_hint, scopes, claims, request):
    """Ensure client supplied user id hint matches session user.

        If the sub claim or id_token_hint is supplied then the session
        user must match the given ID.

        :param id_token_hint: User identifier string.
        :param scopes: List of OAuth 2 scopes and OpenID claims (strings).
        :param claims: OpenID Connect claims dict.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - OpenIDConnectAuthCode
            - OpenIDConnectImplicit
            - OpenIDConnectHybrid
        """
    raise NotImplementedError('Subclasses must implement this method.')