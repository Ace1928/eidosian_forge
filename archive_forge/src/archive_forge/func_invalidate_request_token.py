from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def invalidate_request_token(self, client_key, request_token, request):
    """Invalidates a used request token.

        :param client_key: The client/consumer key.
        :param request_token: The request token string.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: None

        Per `Section 2.3`__ of the spec:

        "The server MUST (...) ensure that the temporary
        credentials have not expired or been used before."

        .. _`Section 2.3`: https://tools.ietf.org/html/rfc5849#section-2.3

        This method should ensure that provided token won't validate anymore.
        It can be simply removing RequestToken from storage or setting
        specific flag that makes it invalid (note that such flag should be
        also validated during request token validation).

        This method is used by

        * AccessTokenEndpoint
        """
    raise self._subclass_must_implement('invalidate_request_token')