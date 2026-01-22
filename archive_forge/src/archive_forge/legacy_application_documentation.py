from __future__ import absolute_import, unicode_literals
from ..parameters import parse_token_response, prepare_token_request
from .base import Client
Add the resource owner password and username to the request body.

        The client makes a request to the token endpoint by adding the
        following parameters using the "application/x-www-form-urlencoded"
        format per `Appendix B`_ in the HTTP request entity-body:

        :param username:    The resource owner username.
        :param password:    The resource owner password.
        :param body: Existing request body (URL encoded string) to embed
        parameters
                     into. This may contain extra paramters. Default ''.
        :param scope:   The scope of the access request as described by
                        `Section 3.3`_.
        :param include_client_id: `True` to send the `client_id` in the body of
                                  the upstream request. Default `None`. This is
                                  required if the client is not authenticating
                                  with the authorization server as described
                                  in `Section 3.2.1`_.
        :type include_client_id: Boolean
        :param kwargs:  Extra credentials to include in the token request.

        If the client type is confidential or the client was issued client
        credentials (or assigned other authentication requirements), the
        client MUST authenticate with the authorization server as described
        in `Section 3.2.1`_.

        The prepared body will include all provided credentials as well as
        the ``grant_type`` parameter set to ``password``::

            >>> from oauthlib.oauth2 import LegacyApplicationClient
            >>> client = LegacyApplicationClient('your_id')
            >>> client.prepare_request_body(username='foo', password='bar',
            scope=['hello', 'world'])
            'grant_type=password&username=foo&scope=hello+world&password=bar'

        .. _`Appendix B`: https://tools.ietf.org/html/rfc6749#appendix-B
        .. _`Section 3.3`: https://tools.ietf.org/html/rfc6749#section-3.3
        .. _`Section 3.2.1`: https://tools.ietf.org/html/rfc6749#section-3.2.1
        