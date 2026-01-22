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
def process_only_nodes(document: Node, tags: 'Tags') -> None:
    """Filter ``only`` nodes which do not match *tags*."""
    for node in document.findall(addnodes.only):
        try:
            ret = tags.eval_condition(node['expr'])
        except Exception as err:
            logger.warning(__('exception while evaluating only directive expression: %s'), err, location=node)
            node.replace_self(node.children or nodes.comment())
        else:
            if ret:
                node.replace_self(node.children or nodes.comment())
            else:
                node.replace_self(nodes.comment())