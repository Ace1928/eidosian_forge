import logging
import httplib2
import six
from six.moves import http_client
from oauth2client import _helpers
def wrap_http_for_auth(credentials, http):
    """Prepares an HTTP object's request method for auth.

    Wraps HTTP requests with logic to catch auth failures (typically
    identified via a 401 status code). In the event of failure, tries
    to refresh the token used and then retry the original request.

    Args:
        credentials: Credentials, the credentials used to identify
                     the authenticated user.
        http: httplib2.Http, an http object to be used to make
              auth requests.
    """
    orig_request_method = http.request

    def new_request(uri, method='GET', body=None, headers=None, redirections=httplib2.DEFAULT_MAX_REDIRECTS, connection_type=None):
        if not credentials.access_token:
            _LOGGER.info('Attempting refresh to obtain initial access_token')
            credentials._refresh(orig_request_method)
        headers = _initialize_headers(headers)
        credentials.apply(headers)
        _apply_user_agent(headers, credentials.user_agent)
        body_stream_position = None
        if all((getattr(body, stream_prop, None) for stream_prop in _STREAM_PROPERTIES)):
            body_stream_position = body.tell()
        resp, content = request(orig_request_method, uri, method, body, clean_headers(headers), redirections, connection_type)
        max_refresh_attempts = 2
        for refresh_attempt in range(max_refresh_attempts):
            if resp.status not in REFRESH_STATUS_CODES:
                break
            _LOGGER.info('Refreshing due to a %s (attempt %s/%s)', resp.status, refresh_attempt + 1, max_refresh_attempts)
            credentials._refresh(orig_request_method)
            credentials.apply(headers)
            if body_stream_position is not None:
                body.seek(body_stream_position)
            resp, content = request(orig_request_method, uri, method, body, clean_headers(headers), redirections, connection_type)
        return (resp, content)
    http.request = new_request
    http.request.credentials = credentials