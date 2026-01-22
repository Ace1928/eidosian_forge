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
def unregister_node(node: Type[Element]) -> None:
    """Unregister a node from docutils.

    This is inverse of ``nodes._add_nodes_class_names()``.
    """
    if hasattr(nodes.GenericNodeVisitor, 'visit_' + node.__name__):
        delattr(nodes.GenericNodeVisitor, 'visit_' + node.__name__)
        delattr(nodes.GenericNodeVisitor, 'depart_' + node.__name__)
        delattr(nodes.SparseNodeVisitor, 'visit_' + node.__name__)
        delattr(nodes.SparseNodeVisitor, 'depart_' + node.__name__)