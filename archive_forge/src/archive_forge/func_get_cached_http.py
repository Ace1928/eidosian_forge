import logging
import httplib2
import six
from six.moves import http_client
from oauth2client import _helpers
def get_cached_http():
    """Return an HTTP object which caches results returned.

    This is intended to be used in methods like
    oauth2client.client.verify_id_token(), which calls to the same URI
    to retrieve certs.

    Returns:
        httplib2.Http, an HTTP object with a MemoryCache
    """
    return _CACHED_HTTP