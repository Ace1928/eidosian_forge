import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def rewrite_links(self, link_repl_func, resolve_base_href=True, base_href=None):
    """
        Rewrite all the links in the document.  For each link
        ``link_repl_func(link)`` will be called, and the return value
        will replace the old link.

        Note that links may not be absolute (unless you first called
        ``make_links_absolute()``), and may be internal (e.g.,
        ``'#anchor'``).  They can also be values like
        ``'mailto:email'`` or ``'javascript:expr'``.

        If you give ``base_href`` then all links passed to
        ``link_repl_func()`` will take that into account.

        If the ``link_repl_func`` returns None, the attribute or
        tag text will be removed completely.
        """
    if base_href is not None:
        self.make_links_absolute(base_href, resolve_base_href=resolve_base_href)
    elif resolve_base_href:
        self.resolve_base_href()
    for el, attrib, link, pos in self.iterlinks():
        new_link = link_repl_func(link.strip())
        if new_link == link:
            continue
        if new_link is None:
            if attrib is None:
                el.text = ''
            else:
                del el.attrib[attrib]
            continue
        if attrib is None:
            new = el.text[:pos] + new_link + el.text[pos + len(link):]
            el.text = new
        else:
            cur = el.get(attrib)
            if not pos and len(cur) == len(link):
                new = new_link
            else:
                new = cur[:pos] + new_link + cur[pos + len(link):]
            el.set(attrib, new)