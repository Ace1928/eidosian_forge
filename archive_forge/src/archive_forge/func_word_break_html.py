import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def word_break_html(html, *args, **kw):
    result_type = type(html)
    doc = fromstring(html)
    word_break(doc, *args, **kw)
    return _transform_result(result_type, doc)