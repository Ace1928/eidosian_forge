from __future__ import absolute_import, unicode_literals
import logging
def client_authentication_required(self, request, *args, **kwargs):
    """Determine if client authentication is required for current request.

        According to the rfc6749, client authentication is required in the
        following cases:
            - Resource Owner Password Credentials Grant, when Client type is
            Confidential or when
              Client was issued client credentials or whenever Client provided
              client
              authentication, see `Section 4.3.2`_.
            - Authorization Code Grant, when Client type is Confidential or when
            Client was issued
              client credentials or whenever Client provided client
              authentication,
              see `Section 4.1.3`_.
            - Refresh Token Grant, when Client type is Confidential or when
            Client was issued
              client credentials or whenever Client provided client
              authentication, see
              `Section 6`_

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - Authorization Code Grant
            - Resource Owner Password Credentials Grant
            - Refresh Token Grant

        .. _`Section 4.3.2`: https://tools.ietf.org/html/rfc6749#section-4.3.2
        .. _`Section 4.1.3`: https://tools.ietf.org/html/rfc6749#section-4.1.3
        .. _`Section 6`: https://tools.ietf.org/html/rfc6749#section-6
        """
    return True