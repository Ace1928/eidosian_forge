import os
import re
import warnings
from contextlib import contextmanager
from copy import copy
from os import path
from types import ModuleType
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set,
import docutils
from docutils import nodes
from docutils.io import FileOutput
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives, roles
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import State, StateMachine, StringList
from docutils.utils import Reporter, unescape
from docutils.writers._html_base import HTMLTranslator
from sphinx.deprecation import RemovedInSphinx70Warning, deprecated_alias
from sphinx.errors import SphinxError
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.typing import RoleFunction
@contextmanager
def du19_footnotes() -> Generator[None, None, None]:

    def visit_footnote(self, node):
        label_style = self.settings.footnote_references
        if not isinstance(node.previous_sibling(), type(node)):
            self.body.append(f'<aside class="footnote-list {label_style}">\n')
        self.body.append(self.starttag(node, 'aside', classes=[node.tagname, label_style], role='note'))

    def depart_footnote(self, node):
        self.body.append('</aside>\n')
        if not isinstance(node.next_node(descend=False, siblings=True), type(node)):
            self.body.append('</aside>\n')
    old_visit_footnote = HTMLTranslator.visit_footnote
    old_depart_footnote = HTMLTranslator.depart_footnote
    if docutils.__version_info__[:2] == (0, 18):
        HTMLTranslator.visit_footnote = visit_footnote
        HTMLTranslator.depart_footnote = depart_footnote
    try:
        yield
    finally:
        if docutils.__version_info__[:2] == (0, 18):
            HTMLTranslator.visit_footnote = old_visit_footnote
            HTMLTranslator.depart_footnote = old_depart_footnote