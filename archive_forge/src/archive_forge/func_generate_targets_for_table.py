import os
import posixpath
import re
import urllib.parse
import warnings
from typing import TYPE_CHECKING, Iterable, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node, Text
from docutils.writers.html5_polyglot import HTMLTranslator as BaseTranslator
from sphinx import addnodes
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.images import get_image_size
def generate_targets_for_table(self, node: Element) -> None:
    """Generate hyperlink targets for tables.

        Original visit_table() generates hyperlink targets inside table tags
        (<table>) if multiple IDs are assigned to listings.
        That is invalid DOM structure.  (This is a bug of docutils <= 0.13.1)

        This exports hyperlink targets before tables to make valid DOM structure.
        """
    warnings.warn('generate_targets_for_table() is deprecated', RemovedInSphinx60Warning, stacklevel=2)
    for id in node['ids'][1:]:
        self.body.append('<span id="%s"></span>' % id)
        node['ids'].remove(id)