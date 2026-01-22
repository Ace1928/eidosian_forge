from __future__ import absolute_import, unicode_literals
import logging
def save_authorization_code(self, client_id, code, request, *args, **kwargs):
    """Persist the authorization_code.

        The code should at minimum be stored with:
            - the client_id (``client_id``)
            - the redirect URI used (``request.redirect_uri``)
            - a resource owner / user (``request.user``)
            - the authorized scopes (``request.scopes``)
            - the client state, if given (``code.get('state')``)

        To support PKCE, you MUST associate the code with:
            - Code Challenge (``request.code_challenge``) and
            - Code Challenge Method (``request.code_challenge_method``)

        The ``code`` argument is actually a dictionary, containing at least a
        ``code`` key with the actual authorization code:

            ``{'code': 'sdf345jsdf0934f'}``

        It may also have a ``state`` key containing a nonce for the client, if
        it
        chose to send one.  That value should be saved and used in
        ``.validate_code``.

        It may also have a ``claims`` parameter which, when present, will be a
        dict
        deserialized from JSON as described at
        http://openid.net/specs/openid-connect-core-1_0.html#ClaimsParameter
        This value should be saved in this method and used again in
        ``.validate_code``.

        :param client_id: Unicode client identifier.
        :param code: A dict of the authorization code grant and, optionally,
        state.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request

        Method is used by:
            - Authorization Code Grant
        """
    raise NotImplementedError('Subclasses must implement this method.')