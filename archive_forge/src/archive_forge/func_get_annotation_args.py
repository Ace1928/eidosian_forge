from __future__ import annotations
import inspect
import re
import sys
import textwrap
import types
from ast import FunctionDef, Module, stmt
from dataclasses import dataclass
from typing import Any, AnyStr, Callable, ForwardRef, NewType, TypeVar, get_type_hints
from docutils.frontend import OptionParser
from docutils.nodes import Node
from docutils.parsers.rst import Parser as RstParser
from docutils.utils import new_document
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import Options
from sphinx.ext.autodoc.mock import mock
from sphinx.util import logging
from sphinx.util.inspect import signature as sphinx_signature
from sphinx.util.inspect import stringify_signature
from .patches import install_patches
from .version import __version__
def get_annotation_args(annotation: Any, module: str, class_name: str) -> tuple[Any, ...]:
    try:
        original = getattr(sys.modules[module], class_name)
    except (KeyError, AttributeError):
        pass
    else:
        if annotation is original:
            return ()
    if class_name in ('Pattern', 'Match') and hasattr(annotation, 'type_var'):
        return (annotation.type_var,)
    elif class_name == 'ClassVar' and hasattr(annotation, '__type__'):
        return (annotation.__type__,)
    elif class_name == 'TypeVar' and hasattr(annotation, '__constraints__'):
        return annotation.__constraints__
    elif class_name == 'NewType' and hasattr(annotation, '__supertype__'):
        return (annotation.__supertype__,)
    elif class_name == 'Literal' and hasattr(annotation, '__values__'):
        return annotation.__values__
    elif class_name == 'Generic':
        return annotation.__parameters__
    result = getattr(annotation, '__args__', ())
    result = () if len(result) == 1 and result[0] == () else result
    return result