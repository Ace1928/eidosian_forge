import functools
import email.utils
import re
import builtins
from binascii import b2a_base64
from cgi import parse_header
from email.header import decode_header
from http.server import BaseHTTPRequestHandler
from urllib.parse import unquote_plus
import jaraco.collections
import cherrypy
from cherrypy._cpcompat import ntob, ntou
def urljoin_bytes(*atoms):
    """Return the given path `*atoms`, joined into a single URL.

    This will correctly join a SCRIPT_NAME and PATH_INFO into the
    original URL, even if either atom is blank.
    """
    url = b'/'.join([x for x in atoms if x])
    while b'//' in url:
        url = url.replace(b'//', b'/')
    return url or b'/'