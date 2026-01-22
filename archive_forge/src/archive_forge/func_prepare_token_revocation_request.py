from __future__ import absolute_import, unicode_literals
import time
import warnings
from oauthlib.common import generate_token
from oauthlib.oauth2.rfc6749 import tokens
from oauthlib.oauth2.rfc6749.errors import (InsecureTransportError,
from oauthlib.oauth2.rfc6749.parameters import (parse_token_response,
from oauthlib.oauth2.rfc6749.utils import is_secure_transport
def prepare_token_revocation_request(self, revocation_url, token, token_type_hint='access_token', body='', callback=None, **kwargs):
    """Prepare a token revocation request.

        :param revocation_url: Provider token revocation endpoint URL.

        :param token: The access or refresh token to be revoked (string).

        :param token_type_hint: ``"access_token"`` (default) or
        ``"refresh_token"``. This is optional and if you wish to not pass it you
        must provide ``token_type_hint=None``.

        :param callback: A jsonp callback such as ``package.callback`` to be
        invoked
        upon receiving the response. Not that it should not include a () suffix.

        :param kwargs: Additional parameters to included in the request.

        :returns: The prepared request tuple with (url, headers, body).

        Note that JSONP request may use GET requests as the parameters will
        be added to the request URL query as opposed to the request body.

        An example of a revocation request

        .. code-block: http

            POST /revoke HTTP/1.1
            Host: server.example.com
            Content-Type: application/x-www-form-urlencoded
            Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW

            token=45ghiukldjahdnhzdauz&token_type_hint=refresh_token

        An example of a jsonp revocation request

        .. code-block: http

            GET /revoke?token=agabcdefddddafdd&callback=package.myCallback
            HTTP/1.1
            Host: server.example.com
            Content-Type: application/x-www-form-urlencoded
            Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW

        and an error response

        .. code-block: http

        package.myCallback({"error":"unsupported_token_type"});

        Note that these requests usually require client credentials, client_id
        in
        the case for public clients and provider specific authentication
        credentials for confidential clients.
        """
    if not is_secure_transport(revocation_url):
        raise InsecureTransportError()
    return prepare_token_revocation_request(revocation_url, token, token_type_hint=token_type_hint, body=body, callback=callback, **kwargs)