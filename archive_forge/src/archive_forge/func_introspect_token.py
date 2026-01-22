from __future__ import absolute_import, unicode_literals
import logging
def introspect_token(self, token, token_type_hint, request, *args, **kwargs):
    """Introspect an access or refresh token.

        Called once the introspect request is validated. This method should
        verify the *token* and either return a dictionary with the list of
        claims associated, or `None` in case the token is unknown.

        Below the list of registered claims you should be interested in:
        - scope : space-separated list of scopes
        - client_id : client identifier
        - username : human-readable identifier for the resource owner
        - token_type : type of the token
        - exp : integer timestamp indicating when this token will expire
        - iat : integer timestamp indicating when this token was issued
        - nbf : integer timestamp indicating when it can be "not-before" used
        - sub : subject of the token - identifier of the resource owner
        - aud : list of string identifiers representing the intended audience
        - iss : string representing issuer of this token
        - jti : string identifier for the token

        Note that most of them are coming directly from JWT RFC. More details
        can be found in `Introspect Claims`_ or `_JWT Claims`_.

        The implementation can use *token_type_hint* to improve lookup
        efficency, but must fallback to other types to be compliant with RFC.

        The dict of claims is added to request.token after this method.

        :param token: The token string.
        :param token_type_hint: access_token or refresh_token.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request

        Method is used by:
            - Introspect Endpoint (all grants are compatible)

        .. _`Introspect Claims`: https://tools.ietf.org/html/rfc7662#section-2.2
        .. _`JWT Claims`: https://tools.ietf.org/html/rfc7519#section-4
        """
    raise NotImplementedError('Subclasses must implement this method.')