from __future__ import absolute_import, unicode_literals
import logging
def validate_redirect_uri(self, client_id, redirect_uri, request, *args, **kwargs):
    """Ensure client is authorized to redirect to the redirect_uri requested.

        All clients should register the absolute URIs of all URIs they intend
        to redirect to. The registration is outside of the scope of oauthlib.

        :param client_id: Unicode client identifier.
        :param redirect_uri: Unicode absolute URI.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - Authorization Code Grant
            - Implicit Grant
        """
    raise NotImplementedError('Subclasses must implement this method.')