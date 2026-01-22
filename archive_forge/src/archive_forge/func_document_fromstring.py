import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def document_fromstring(html, parser=None, ensure_head_body=False, **kw):
    if parser is None:
        parser = html_parser
    value = etree.fromstring(html, parser, **kw)
    if value is None:
        raise etree.ParserError('Document is empty')
    if ensure_head_body and value.find('head') is None:
        value.insert(0, Element('head'))
    if ensure_head_body and value.find('body') is None:
        value.append(Element('body'))
    return value