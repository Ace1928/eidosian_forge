import logging
import httplib2
import six
from six.moves import http_client
from oauth2client import _helpers
def get_http_object(*args, **kwargs):
    """Return a new HTTP object.

    Args:
        *args: tuple, The positional arguments to be passed when
               contructing a new HTTP object.
        **kwargs: dict, The keyword arguments to be passed when
                  contructing a new HTTP object.

    Returns:
        httplib2.Http, an HTTP object.
    """
    return httplib2.Http(*args, **kwargs)