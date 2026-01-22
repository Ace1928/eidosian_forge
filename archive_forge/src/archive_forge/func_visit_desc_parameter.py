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
def visit_desc_parameter(self, node: Element) -> None:
    if self.first_param:
        self.first_param = 0
    elif not self.required_params_left:
        self.body.append(self.param_separator)
    if self.optional_param_level == 0:
        self.required_params_left -= 1
    if not node.hasattr('noemph'):
        self.body.append('<em>')