from __future__ import absolute_import, unicode_literals
import datetime
import os
from oauthlib.common import unicode_type, urldecode
def host_from_uri(uri):
    """Extract hostname and port from URI.

    Will use default port for HTTP and HTTPS if none is present in the URI.
    """
    default_ports = {'HTTP': '80', 'HTTPS': '443'}
    sch, netloc, path, par, query, fra = urlparse(uri)
    if ':' in netloc:
        netloc, port = netloc.split(':', 1)
    else:
        port = default_ports.get(sch.upper())
    return (netloc, port)