import re
import textwrap
from os import path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Pattern, Set,
from docutils import nodes, writers
from docutils.nodes import Element, Node, Text
from sphinx import __display_version__, addnodes
from sphinx.domains import IndexEntry
from sphinx.domains.index import IndexDomain
from sphinx.errors import ExtensionError
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.i18n import format_date
from sphinx.writers.latex import collected_footnote
def tex_image_length(self, width_str: str) -> str:
    match = re.match('(\\d*\\.?\\d*)\\s*(\\S*)', width_str)
    if not match:
        return width_str
    res = width_str
    amount, unit = match.groups()[:2]
    if not unit or unit == 'px':
        return ''
    elif unit == '%':
        res = '%d.0pt' % (float(amount) * 4.1825368)
    return res