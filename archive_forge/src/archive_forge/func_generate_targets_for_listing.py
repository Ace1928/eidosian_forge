import os
import posixpath
import re
import urllib.parse
import warnings
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node, Text
from docutils.writers.html4css1 import HTMLTranslator as BaseTranslator
from docutils.writers.html4css1 import Writer
from sphinx import addnodes
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.images import get_image_size
def generate_targets_for_listing(self, node: Element) -> None:
    """Generate hyperlink targets for listings.

        Original visit_bullet_list(), visit_definition_list() and visit_enumerated_list()
        generates hyperlink targets inside listing tags (<ul>, <ol> and <dl>) if multiple
        IDs are assigned to listings.  That is invalid DOM structure.
        (This is a bug of docutils <= 0.12)

        This exports hyperlink targets before listings to make valid DOM structure.
        """
    for id in node['ids'][1:]:
        self.body.append('<span id="%s"></span>' % id)
        node['ids'].remove(id)