from __future__ import annotations
import html
import os
import posixpath
import re
from collections.abc import Iterable
from os import path
from typing import Any, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import canon_path, make_filename
from sphinx.util.template import SphinxRenderer
def keyword_item(self, name: str, ref: Any) -> str:
    matchobj = _idpattern.match(name)
    if matchobj:
        groupdict = matchobj.groupdict()
        shortname = groupdict['title']
        id = groupdict.get('id')
        if shortname.endswith('()'):
            shortname = shortname[:-2]
        id = html.escape('%s.%s' % (id, shortname), True)
    else:
        id = None
    nameattr = html.escape(name, quote=True)
    refattr = html.escape(ref[1], quote=True)
    if id:
        item = ' ' * 12 + '<keyword name="%s" id="%s" ref="%s"/>' % (nameattr, id, refattr)
    else:
        item = ' ' * 12 + '<keyword name="%s" ref="%s"/>' % (nameattr, refattr)
    item.encode('ascii', 'xmlcharrefreplace')
    return item