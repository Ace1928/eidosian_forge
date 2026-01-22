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
def protocol_from_http(protocol_str):
    """Return a protocol tuple from the given 'HTTP/x.y' string."""
    return (int(protocol_str[5]), int(protocol_str[7]))