from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import Request
from ..errors import OAuth2Error, UnsupportedTokenTypeError
from .base import BaseEndpoint, catch_errors_and_unavailability
def validate_revocation_request(self, request):
    """Ensure the request is valid.

        The client constructs the request by including the following parameters
        using the "application/x-www-form-urlencoded" format in the HTTP
        request entity-body:

        token (REQUIRED).  The token that the client wants to get revoked.

        token_type_hint (OPTIONAL).  A hint about the type of the token
        submitted for revocation.  Clients MAY pass this parameter in order to
        help the authorization server to optimize the token lookup.  If the
        server is unable to locate the token using the given hint, it MUST
        extend its search accross all of its supported token types.  An
        authorization server MAY ignore this parameter, particularly if it is
        able to detect the token type automatically.  This specification
        defines two such values:

                *  access_token: An Access Token as defined in [RFC6749],
                    `section 1.4`_

                *  refresh_token: A Refresh Token as defined in [RFC6749],
                    `section 1.5`_

                Specific implementations, profiles, and extensions of this
                specification MAY define other values for this parameter using
                the registry defined in `Section 4.1.2`_.

        The client also includes its authentication credentials as described in
        `Section 2.3`_. of [`RFC6749`_].

        .. _`section 1.4`: https://tools.ietf.org/html/rfc6749#section-1.4
        .. _`section 1.5`: https://tools.ietf.org/html/rfc6749#section-1.5
        .. _`section 2.3`: https://tools.ietf.org/html/rfc6749#section-2.3
        .. _`Section 4.1.2`:
        https://tools.ietf.org/html/draft-ietf-oauth-revocation-11#section-4.1.2
        .. _`RFC6749`: https://tools.ietf.org/html/rfc6749
        """
    self._raise_on_missing_token(request)
    self._raise_on_invalid_client(request)
    self._raise_on_unsupported_token(request)