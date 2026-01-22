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
def lookup_domain_element(self, type: str, name: str) -> Any:
    """Lookup a markup element (directive or role), given its name which can
        be a full name (with domain).
        """
    name = name.lower()
    if ':' in name:
        domain_name, name = name.split(':', 1)
        if domain_name in self.env.domains:
            domain = self.env.get_domain(domain_name)
            element = getattr(domain, type)(name)
            if element is not None:
                return (element, [])
        else:
            logger.warning(_('unknown directive or role name: %s:%s'), domain_name, name)
    else:
        def_domain = self.env.temp_data.get('default_domain')
        if def_domain is not None:
            element = getattr(def_domain, type)(name)
            if element is not None:
                return (element, [])
    element = getattr(self.env.get_domain('std'), type)(name)
    if element is not None:
        return (element, [])
    raise ElementLookupError