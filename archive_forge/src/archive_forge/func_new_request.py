import logging
import httplib2
import six
from six.moves import http_client
from oauth2client import _helpers
def new_request(uri, method='GET', body=None, headers=None, redirections=httplib2.DEFAULT_MAX_REDIRECTS, connection_type=None):
    if 'aud' in credentials._kwargs:
        if credentials.access_token is None or credentials.access_token_expired:
            credentials.refresh(None)
        return request(authenticated_request_method, uri, method, body, headers, redirections, connection_type)
    else:
        headers = _initialize_headers(headers)
        _apply_user_agent(headers, credentials.user_agent)
        uri_root = uri.split('?', 1)[0]
        token, unused_expiry = credentials._create_token({'aud': uri_root})
        headers['Authorization'] = 'Bearer ' + token
        return request(orig_request_method, uri, method, body, clean_headers(headers), redirections, connection_type)