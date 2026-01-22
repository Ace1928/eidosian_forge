import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def tostring(doc, pretty_print=False, include_meta_content_type=False, encoding=None, method='html', with_tail=True, doctype=None):
    """Return an HTML string representation of the document.

    Note: if include_meta_content_type is true this will create a
    ``<meta http-equiv="Content-Type" ...>`` tag in the head;
    regardless of the value of include_meta_content_type any existing
    ``<meta http-equiv="Content-Type" ...>`` tag will be removed

    The ``encoding`` argument controls the output encoding (defaults to
    ASCII, with &#...; character references for any characters outside
    of ASCII).  Note that you can pass the name ``'unicode'`` as
    ``encoding`` argument to serialise to a Unicode string.

    The ``method`` argument defines the output method.  It defaults to
    'html', but can also be 'xml' for xhtml output, or 'text' to
    serialise to plain text without markup.

    To leave out the tail text of the top-level element that is being
    serialised, pass ``with_tail=False``.

    The ``doctype`` option allows passing in a plain string that will
    be serialised before the XML tree.  Note that passing in non
    well-formed content here will make the XML output non well-formed.
    Also, an existing doctype in the document tree will not be removed
    when serialising an ElementTree instance.

    Example::

        >>> from lxml import html
        >>> root = html.fragment_fromstring('<p>Hello<br>world!</p>')

        >>> html.tostring(root)
        b'<p>Hello<br>world!</p>'
        >>> html.tostring(root, method='html')
        b'<p>Hello<br>world!</p>'

        >>> html.tostring(root, method='xml')
        b'<p>Hello<br/>world!</p>'

        >>> html.tostring(root, method='text')
        b'Helloworld!'

        >>> html.tostring(root, method='text', encoding='unicode')
        u'Helloworld!'

        >>> root = html.fragment_fromstring('<div><p>Hello<br>world!</p>TAIL</div>')
        >>> html.tostring(root[0], method='text', encoding='unicode')
        u'Helloworld!TAIL'

        >>> html.tostring(root[0], method='text', encoding='unicode', with_tail=False)
        u'Helloworld!'

        >>> doc = html.document_fromstring('<p>Hello<br>world!</p>')
        >>> html.tostring(doc, method='html', encoding='unicode')
        u'<html><body><p>Hello<br>world!</p></body></html>'

        >>> print(html.tostring(doc, method='html', encoding='unicode',
        ...          doctype='<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"'
        ...                  ' "http://www.w3.org/TR/html4/strict.dtd">'))
        <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
        <html><body><p>Hello<br>world!</p></body></html>
    """
    html = etree.tostring(doc, method=method, pretty_print=pretty_print, encoding=encoding, with_tail=with_tail, doctype=doctype)
    if method == 'html' and (not include_meta_content_type):
        if isinstance(html, str):
            html = __str_replace_meta_content_type('', html)
        else:
            html = __bytes_replace_meta_content_type(b'', html)
    return html