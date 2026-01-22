import base64
import binascii
import copy
import html.entities
import re
import xml.sax.saxutils
from .html import _cp1252
from .namespaces import _base, cc, dc, georss, itunes, mediarss, psc
from .sanitizer import _sanitize_html, _HTMLSanitizer
from .util import FeedParserDict
from .urls import _urljoin, make_safe_absolute_uri, resolve_relative_uris
@staticmethod
def map_content_type(content_type):
    content_type = content_type.lower()
    if content_type == 'text' or content_type == 'plain':
        content_type = 'text/plain'
    elif content_type == 'html':
        content_type = 'text/html'
    elif content_type == 'xhtml':
        content_type = 'application/xhtml+xml'
    return content_type