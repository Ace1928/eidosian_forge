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
def is_smartquotable(node: Node) -> bool:
    """Check whether the node is smart-quotable or not."""
    for pnode in traverse_parent(node.parent):
        if isinstance(pnode, NON_SMARTQUOTABLE_PARENT_NODES):
            return False
        elif pnode.get('support_smartquotes', None) is False:
            return False
    if getattr(node, 'support_smartquotes', None) is False:
        return False
    return True