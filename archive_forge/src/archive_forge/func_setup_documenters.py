import argparse
import inspect
import locale
import os
import pkgutil
import pydoc
import re
import sys
from gettext import NullTranslations
from os import path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Type
from jinja2 import TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.ext.autodoc import Documenter
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autosummary import (ImportExceptionGroup, get_documenter, import_by_name,
from sphinx.locale import __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.registry import SphinxComponentRegistry
from sphinx.util import logging, rst, split_full_qualified_name
from sphinx.util.inspect import getall, safe_getattr
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxTemplateLoader
def setup_documenters(app: Any) -> None:
    from sphinx.ext.autodoc import AttributeDocumenter, ClassDocumenter, DataDocumenter, DecoratorDocumenter, ExceptionDocumenter, FunctionDocumenter, MethodDocumenter, ModuleDocumenter, NewTypeAttributeDocumenter, NewTypeDataDocumenter, PropertyDocumenter
    documenters: List[Type[Documenter]] = [ModuleDocumenter, ClassDocumenter, ExceptionDocumenter, DataDocumenter, FunctionDocumenter, MethodDocumenter, NewTypeAttributeDocumenter, NewTypeDataDocumenter, AttributeDocumenter, DecoratorDocumenter, PropertyDocumenter]
    for documenter in documenters:
        app.registry.add_documenter(documenter.objtype, documenter)