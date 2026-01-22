import io
import sys
import six
import types
from six import StringIO
from io import BytesIO
from lxml import etree
from ncclient import NCClientError
def to_ele(x, huge_tree=False):
    """Convert and return the :class:`~xml.etree.ElementTree.Element` for the XML document *x*. If *x* is already an :class:`~xml.etree.ElementTree.Element` simply returns that.

    *huge_tree*: parse XML with very deep trees and very long text content
    """
    if sys.version < '3':
        return x if etree.iselement(x) else etree.fromstring(x, parser=_get_parser(huge_tree))
    else:
        return x if etree.iselement(x) else etree.fromstring(x.encode('UTF-8'), parser=_get_parser(huge_tree))