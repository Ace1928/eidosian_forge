import inspect
import os
import posixpath
import re
import sys
import warnings
from inspect import Parameter
from os import path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Node, system_message
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import RSTStateMachine, Struct, state_classes
from docutils.statemachine import StringList
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.deprecation import (RemovedInSphinx60Warning, RemovedInSphinx70Warning,
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import INSTANCEATTR, Documenter
from sphinx.ext.autodoc.directive import DocumenterBridge, Options
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autodoc.mock import mock
from sphinx.extension import Extension
from sphinx.locale import __
from sphinx.project import Project
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.registry import SphinxComponentRegistry
from sphinx.util import logging, rst
from sphinx.util.docutils import (NullReporter, SphinxDirective, SphinxRole, new_document,
from sphinx.util.inspect import signature_from_str
from sphinx.util.matching import Matcher
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
def get_documenter(app: Sphinx, obj: Any, parent: Any) -> Type[Documenter]:
    """Get an autodoc.Documenter class suitable for documenting the given
    object.

    *obj* is the Python object to be documented, and *parent* is an
    another Python object (e.g. a module or a class) to which *obj*
    belongs to.
    """
    from sphinx.ext.autodoc import DataDocumenter, ModuleDocumenter
    if inspect.ismodule(obj):
        return ModuleDocumenter
    if parent is not None:
        parent_doc_cls = get_documenter(app, parent, None)
    else:
        parent_doc_cls = ModuleDocumenter
    if hasattr(parent, '__name__'):
        parent_doc = parent_doc_cls(FakeDirective(), parent.__name__)
    else:
        parent_doc = parent_doc_cls(FakeDirective(), '')
    classes = [cls for cls in app.registry.documenters.values() if cls.can_document_member(obj, '', False, parent_doc)]
    if classes:
        classes.sort(key=lambda cls: cls.priority)
        return classes[-1]
    else:
        return DataDocumenter