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
def find_autosummary_in_lines(lines: List[str], module: Optional[str]=None, filename: Optional[str]=None) -> List[AutosummaryEntry]:
    """Find out what items appear in autosummary:: directives in the
    given lines.

    Returns a list of (name, toctree, template) where *name* is a name
    of an object and *toctree* the :toctree: path of the corresponding
    autosummary directive (relative to the root of the file name), and
    *template* the value of the :template: option. *toctree* and
    *template* ``None`` if the directive does not have the
    corresponding options set.
    """
    autosummary_re = re.compile('^(\\s*)\\.\\.\\s+autosummary::\\s*')
    automodule_re = re.compile('^\\s*\\.\\.\\s+automodule::\\s*([A-Za-z0-9_.]+)\\s*$')
    module_re = re.compile('^\\s*\\.\\.\\s+(current)?module::\\s*([a-zA-Z0-9_.]+)\\s*$')
    autosummary_item_re = re.compile('^\\s+(~?[_a-zA-Z][a-zA-Z0-9_.]*)\\s*.*?')
    recursive_arg_re = re.compile('^\\s+:recursive:\\s*$')
    toctree_arg_re = re.compile('^\\s+:toctree:\\s*(.*?)\\s*$')
    template_arg_re = re.compile('^\\s+:template:\\s*(.*?)\\s*$')
    documented: List[AutosummaryEntry] = []
    recursive = False
    toctree: Optional[str] = None
    template = None
    current_module = module
    in_autosummary = False
    base_indent = ''
    for line in lines:
        if in_autosummary:
            m = recursive_arg_re.match(line)
            if m:
                recursive = True
                continue
            m = toctree_arg_re.match(line)
            if m:
                toctree = m.group(1)
                if filename:
                    toctree = os.path.join(os.path.dirname(filename), toctree)
                continue
            m = template_arg_re.match(line)
            if m:
                template = m.group(1).strip()
                continue
            if line.strip().startswith(':'):
                continue
            m = autosummary_item_re.match(line)
            if m:
                name = m.group(1).strip()
                if name.startswith('~'):
                    name = name[1:]
                if current_module and (not name.startswith(current_module + '.')):
                    name = '%s.%s' % (current_module, name)
                documented.append(AutosummaryEntry(name, toctree, template, recursive))
                continue
            if not line.strip() or line.startswith(base_indent + ' '):
                continue
            in_autosummary = False
        m = autosummary_re.match(line)
        if m:
            in_autosummary = True
            base_indent = m.group(1)
            recursive = False
            toctree = None
            template = None
            continue
        m = automodule_re.search(line)
        if m:
            current_module = m.group(1).strip()
            documented.extend(find_autosummary_in_docstring(current_module, filename=filename))
            continue
        m = module_re.match(line)
        if m:
            current_module = m.group(2)
            continue
    return documented