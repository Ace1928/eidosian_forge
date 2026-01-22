from __future__ import absolute_import, unicode_literals
import logging
def save_bearer_token(self, token, request, *args, **kwargs):
    """Persist the Bearer token.

        The Bearer token should at minimum be associated with:
            - a client and it's client_id, if available
            - a resource owner / user (request.user)
            - authorized scopes (request.scopes)
            - an expiration time
            - a refresh token, if issued
            - a claims document, if present in request.claims

        The Bearer token dict may hold a number of items::

            {
                'token_type': 'Bearer',
                'access_token': 'askfjh234as9sd8',
                'expires_in': 3600,
                'scope': 'string of space separated authorized scopes',
                'refresh_token': '23sdf876234',  # if issued
                'state': 'given_by_client',  # if supplied by client
            }

        Note that while "scope" is a string-separated list of authorized scopes,
        the original list is still available in request.scopes.

        The token dict is passed as a reference so any changes made to the
        dictionary
        will go back to the user.  If additional information must return to the
        client
        user, and it is only possible to get this information after writing the
        token
        to storage, it should be added to the token dictionary.  If the token
        dictionary must be modified but the changes should not go back to the
        user,
        a copy of the dictionary must be made before making the changes.

        Also note that if an Authorization Code grant request included a valid
        claims
        parameter (for OpenID Connect) then the request.claims property will
        contain
        the claims dict, which should be saved for later use when generating the
        id_token and/or UserInfo response content.

        :param token: A Bearer token dict.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: The default redirect URI for the client

        Method is used by all core grant types issuing Bearer tokens:
            - Authorization Code Grant
            - Implicit Grant
            - Resource Owner Password Credentials Grant (might not associate a
            client)
            - Client Credentials grant
        """
    raise NotImplementedError('Subclasses must implement this method.')