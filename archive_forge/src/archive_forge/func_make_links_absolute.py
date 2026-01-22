import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def make_links_absolute(self, base_url=None, resolve_base_href=True, handle_failures=None):
    """
        Make all links in the document absolute, given the
        ``base_url`` for the document (the full URL where the document
        came from), or if no ``base_url`` is given, then the ``.base_url``
        of the document.

        If ``resolve_base_href`` is true, then any ``<base href>``
        tags in the document are used *and* removed from the document.
        If it is false then any such tag is ignored.

        If ``handle_failures`` is None (default), a failure to process
        a URL will abort the processing.  If set to 'ignore', errors
        are ignored.  If set to 'discard', failing URLs will be removed.
        """
    if base_url is None:
        base_url = self.base_url
        if base_url is None:
            raise TypeError('No base_url given, and the document has no base_url')
    if resolve_base_href:
        self.resolve_base_href()
    if handle_failures == 'ignore':

        def link_repl(href):
            try:
                return urljoin(base_url, href)
            except ValueError:
                return href
    elif handle_failures == 'discard':

        def link_repl(href):
            try:
                return urljoin(base_url, href)
            except ValueError:
                return None
    elif handle_failures is None:

        def link_repl(href):
            return urljoin(base_url, href)
    else:
        raise ValueError('unexpected value for handle_failures: %r' % handle_failures)
    self.rewrite_links(link_repl)