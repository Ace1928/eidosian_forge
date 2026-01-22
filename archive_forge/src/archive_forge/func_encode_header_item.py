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
@classmethod
def encode_header_item(cls, item):
    if isinstance(item, str):
        item = cls.encode(item)
    return item.translate(header_translate_table, header_translate_deletechars)