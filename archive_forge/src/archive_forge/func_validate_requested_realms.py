from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def validate_requested_realms(self, client_key, realms, request):
    """Validates that the client may request access to the realm.

        :param client_key: The client/consumer key.
        :param realms: The list of realms that client is requesting access to.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: True or False

        This method is invoked when obtaining a request token and should
        tie a realm to the request token and after user authorization
        this realm restriction should transfer to the access token.

        This method is used by

        * RequestTokenEndpoint
        """
    raise self._subclass_must_implement('validate_requested_realms')