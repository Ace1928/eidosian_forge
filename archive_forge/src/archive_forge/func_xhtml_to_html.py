import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def xhtml_to_html(xhtml):
    """Convert all tags in an XHTML tree to HTML by removing their
    XHTML namespace.
    """
    try:
        xhtml = xhtml.getroot()
    except AttributeError:
        pass
    prefix = '{%s}' % XHTML_NAMESPACE
    prefix_len = len(prefix)
    for el in xhtml.iter(prefix + '*'):
        el.tag = el.tag[prefix_len:]