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
def split_explicit_title(text: str) -> Tuple[bool, str, str]:
    """Split role content into title and target, if given."""
    match = explicit_title_re.match(text)
    if match:
        return (True, match.group(1), match.group(2))
    return (False, text, text)