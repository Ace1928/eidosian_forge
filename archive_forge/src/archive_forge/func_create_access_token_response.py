from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import urlencode
from .. import errors
from .base import BaseEndpoint
def create_access_token_response(self, uri, http_method='GET', body=None, headers=None, credentials=None):
    """Create an access token response, with a new request token if valid.

        :param uri: The full URI of the token request.
        :param http_method: A valid HTTP verb, i.e. GET, POST, PUT, HEAD, etc.
        :param body: The request body as a string.
        :param headers: The request headers as a dict.
        :param credentials: A list of extra credentials to include in the token.
        :returns: A tuple of 3 elements.
                  1. A dict of headers to set on the response.
                  2. The response body as a string.
                  3. The response status code as an integer.

        An example of a valid request::

            >>> from your_validator import your_validator
            >>> from oauthlib.oauth1 import AccessTokenEndpoint
            >>> endpoint = AccessTokenEndpoint(your_validator)
            >>> h, b, s = endpoint.create_access_token_response(
            ...     'https://your.provider/access_token?foo=bar',
            ...     headers={
            ...         'Authorization': 'OAuth oauth_token=234lsdkf....'
            ...     },
            ...     credentials={
            ...         'my_specific': 'argument',
            ...     })
            >>> h
            {'Content-Type': 'application/x-www-form-urlencoded'}
            >>> b
            'oauth_token=lsdkfol23w54jlksdef&oauth_token_secret=qwe089234lkjsdf&oauth_authorized_realms=movies+pics&my_specific=argument'
            >>> s
            200

        An response to invalid request would have a different body and status::

            >>> b
            'error=invalid_request&description=missing+resource+owner+key'
            >>> s
            400

        The same goes for an an unauthorized request:

            >>> b
            ''
            >>> s
            401
        """
    resp_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    try:
        request = self._create_request(uri, http_method, body, headers)
        valid, processed_request = self.validate_access_token_request(request)
        if valid:
            token = self.create_access_token(request, credentials or {})
            self.request_validator.invalidate_request_token(request.client_key, request.resource_owner_key, request)
            return (resp_headers, token, 200)
        else:
            return ({}, None, 401)
    except errors.OAuth1Error as e:
        return (resp_headers, e.urlencoded, e.status_code)