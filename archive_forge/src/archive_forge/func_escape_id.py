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
def escape_id(self, s: str) -> str:
    """Return an escaped string suitable for node names and anchors."""
    bad_chars = ',:()'
    for bc in bad_chars:
        s = s.replace(bc, ' ')
    if re.search('[^ .]', s):
        s = s.replace('.', ' ')
    s = ' '.join(s.split()).strip()
    return self.escape(s)