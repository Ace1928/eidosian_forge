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
def docutils_namespace() -> Generator[None, None, None]:
    """Create namespace for reST parsers."""
    try:
        _directives = copy(directives._directives)
        _roles = copy(roles._roles)
        yield
    finally:
        directives._directives = _directives
        roles._roles = _roles
        for node in list(additional_nodes):
            unregister_node(node)
            additional_nodes.discard(node)