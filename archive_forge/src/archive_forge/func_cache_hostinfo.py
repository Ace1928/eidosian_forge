import os
import re
import six
from six.moves import urllib
from routes import request_config
def cache_hostinfo(environ):
    """Processes the host information and stores a copy

    This work was previously done but wasn't stored in environ, nor is
    it guaranteed to be setup in the future (Routes 2 and beyond).

    cache_hostinfo processes environ keys that may be present to
    determine the proper host, protocol, and port information to use
    when generating routes.

    """
    hostinfo = {}
    if environ.get('HTTPS') or environ.get('wsgi.url_scheme') == 'https' or 'https' in environ.get('HTTP_X_FORWARDED_PROTO', '').split(', '):
        hostinfo['protocol'] = 'https'
    else:
        hostinfo['protocol'] = 'http'
    if environ.get('HTTP_X_FORWARDED_HOST'):
        hostinfo['host'] = environ['HTTP_X_FORWARDED_HOST'].split(', ', 1)[0]
    elif environ.get('HTTP_HOST'):
        hostinfo['host'] = environ['HTTP_HOST']
    else:
        hostinfo['host'] = environ['SERVER_NAME']
        if environ.get('wsgi.url_scheme') == 'https':
            if environ['SERVER_PORT'] != '443':
                hostinfo['host'] += ':' + environ['SERVER_PORT']
        elif environ['SERVER_PORT'] != '80':
            hostinfo['host'] += ':' + environ['SERVER_PORT']
    environ['routes.cached_hostinfo'] = hostinfo
    return hostinfo