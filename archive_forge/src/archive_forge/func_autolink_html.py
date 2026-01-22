import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def autolink_html(html, *args, **kw):
    result_type = type(html)
    if isinstance(html, (str, bytes)):
        doc = fromstring(html)
    else:
        doc = copy.deepcopy(html)
    autolink(doc, *args, **kw)
    return _transform_result(result_type, doc)