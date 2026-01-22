from __future__ import absolute_import, unicode_literals
import time
import warnings
from oauthlib.common import generate_token
from oauthlib.oauth2.rfc6749 import tokens
from oauthlib.oauth2.rfc6749.errors import (InsecureTransportError,
from oauthlib.oauth2.rfc6749.parameters import (parse_token_response,
from oauthlib.oauth2.rfc6749.utils import is_secure_transport
def prepare_refresh_token_request(self, token_url, refresh_token=None, body='', scope=None, **kwargs):
    """Prepare an access token refresh request.

        Expired access tokens can be replaced by new access tokens without
        going through the OAuth dance if the client obtained a refresh token.
        This refresh token and authentication credentials can be used to
        obtain a new access token, and possibly a new refresh token.

        :param token_url: Provider token refresh endpoint URL.

        :param refresh_token: Refresh token string.

        :param body: Existing request body (URL encoded string) to embed
        parameters
                     into. This may contain extra paramters. Default ''.

        :param scope: List of scopes to request. Must be equal to
        or a subset of the scopes granted when obtaining the refresh
        token.

        :param kwargs: Additional parameters to included in the request.

        :returns: The prepared request tuple with (url, headers, body).
        """
    if not is_secure_transport(token_url):
        raise InsecureTransportError()
    self.scope = scope or self.scope
    body = self.prepare_refresh_body(body=body, refresh_token=refresh_token, scope=self.scope, **kwargs)
    return (token_url, FORM_ENC_HEADERS, body)