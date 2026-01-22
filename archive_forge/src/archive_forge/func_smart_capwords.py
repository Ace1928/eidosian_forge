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
def smart_capwords(s: str, sep: Optional[str]=None) -> str:
    """Like string.capwords() but does not capitalize words that already
    contain a capital letter."""
    words = s.split(sep)
    for i, word in enumerate(words):
        if all((x.islower() for x in word)):
            words[i] = word.capitalize()
    return (sep or ' ').join(words)