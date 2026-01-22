import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def fragment_fromstring(html, create_parent=False, base_url=None, parser=None, **kw):
    """
    Parses a single HTML element; it is an error if there is more than
    one element, or if anything but whitespace precedes or follows the
    element.

    If ``create_parent`` is true (or is a tag name) then a parent node
    will be created to encapsulate the HTML in a single element.  In this
    case, leading or trailing text is also allowed, as are multiple elements
    as result of the parsing.

    Passing a ``base_url`` will set the document's ``base_url`` attribute
    (and the tree's docinfo.URL).
    """
    if parser is None:
        parser = html_parser
    accept_leading_text = bool(create_parent)
    elements = fragments_fromstring(html, parser=parser, no_leading_text=not accept_leading_text, base_url=base_url, **kw)
    if create_parent:
        if not isinstance(create_parent, str):
            create_parent = 'div'
        new_root = Element(create_parent)
        if elements:
            if isinstance(elements[0], str):
                new_root.text = elements[0]
                del elements[0]
            new_root.extend(elements)
        return new_root
    if not elements:
        raise etree.ParserError('No elements found')
    if len(elements) > 1:
        raise etree.ParserError('Multiple elements found (%s)' % ', '.join([_element_name(e) for e in elements]))
    el = elements[0]
    if el.tail and el.tail.strip():
        raise etree.ParserError('Element followed by text: %r' % el.tail)
    el.tail = None
    return el