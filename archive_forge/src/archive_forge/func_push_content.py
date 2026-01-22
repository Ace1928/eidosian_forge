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
def push_content(self, tag, attrs_d, default_content_type, expecting_text):
    self.incontent += 1
    if self.lang:
        self.lang = self.lang.replace('_', '-')
    self.contentparams = FeedParserDict({'type': self.map_content_type(attrs_d.get('type', default_content_type)), 'language': self.lang, 'base': self.baseuri})
    self.contentparams['base64'] = self._is_base64(attrs_d, self.contentparams)
    self.push(tag, expecting_text)