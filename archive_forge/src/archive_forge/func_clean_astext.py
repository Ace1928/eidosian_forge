import re
import unicodedata
from typing import (TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Set, Tuple, Type,
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import Directive
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging
def clean_astext(node: Element) -> str:
    """Like node.astext(), but ignore images."""
    node = node.deepcopy()
    for img in node.findall(nodes.image):
        img['alt'] = ''
    for raw in list(node.findall(nodes.raw)):
        raw.parent.remove(raw)
    return node.astext()