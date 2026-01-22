import traceback
import warnings
from importlib import import_module
from types import MethodType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Type,
from docutils import nodes
from docutils.core import Publisher
from docutils.io import Input
from docutils.nodes import Element, Node, TextElement
from docutils.parsers import Parser
from docutils.parsers.rst import Directive
from docutils.transforms import Transform
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning, RemovedInSphinx70Warning
from sphinx.domains import Domain, Index, ObjType
from sphinx.domains.std import GenericObject, Target
from sphinx.environment import BuildEnvironment
from sphinx.errors import ExtensionError, SphinxError, VersionRequirementError
from sphinx.extension import Extension
from sphinx.io import create_publisher
from sphinx.locale import __
from sphinx.parsers import Parser as SphinxParser
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.logging import prefixed_warnings
from sphinx.util.typing import RoleFunction, TitleGetter
def merge_source_suffix(app: 'Sphinx', config: Config) -> None:
    """Merge any user-specified source_suffix with any added by extensions."""
    for suffix, filetype in app.registry.source_suffix.items():
        if suffix not in app.config.source_suffix:
            app.config.source_suffix[suffix] = filetype
        elif app.config.source_suffix[suffix] is None:
            app.config.source_suffix[suffix] = filetype
    app.registry.source_suffix = app.config.source_suffix