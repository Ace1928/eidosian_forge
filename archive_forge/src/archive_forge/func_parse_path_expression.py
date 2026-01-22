import html
import re
import os
from collections.abc import MutableMapping as DictMixin
from paste import httpexceptions
def parse_path_expression(path):
    """
    Parses a path expression like 'domain foobar.com port 20 /' or
    just '/foobar' for a path alone.  Returns as an address that
    URLMap likes.
    """
    parts = path.split()
    domain = port = path = None
    while parts:
        if parts[0] == 'domain':
            parts.pop(0)
            if not parts:
                raise ValueError("'domain' must be followed with a domain name")
            if domain:
                raise ValueError("'domain' given twice")
            domain = parts.pop(0)
        elif parts[0] == 'port':
            parts.pop(0)
            if not parts:
                raise ValueError("'port' must be followed with a port number")
            if port:
                raise ValueError("'port' given twice")
            port = parts.pop(0)
        else:
            if path:
                raise ValueError('more than one path given (have %r, got %r)' % (path, parts[0]))
            path = parts.pop(0)
    s = ''
    if domain:
        s = 'http://%s' % domain
    if port:
        if not domain:
            raise ValueError('If you give a port, you must also give a domain')
        s += ':' + port
    if path:
        if s:
            s += '/'
        s += path
    return s