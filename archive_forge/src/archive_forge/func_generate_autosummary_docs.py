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
def generate_autosummary_docs(sources: List[str], output_dir: Optional[str]=None, suffix: str='.rst', base_path: Optional[str]=None, imported_members: bool=False, app: Any=None, overwrite: bool=True, encoding: str='utf-8') -> None:
    showed_sources = sorted(sources)
    if len(showed_sources) > 20:
        showed_sources = showed_sources[:10] + ['...'] + showed_sources[-10:]
    logger.info(__('[autosummary] generating autosummary for: %s') % ', '.join(showed_sources))
    if output_dir:
        logger.info(__('[autosummary] writing to %s') % output_dir)
    if base_path is not None:
        sources = [os.path.join(base_path, filename) for filename in sources]
    template = AutosummaryRenderer(app)
    items = find_autosummary_in_files(sources)
    new_files = []
    if app:
        filename_map = app.config.autosummary_filename_map
    else:
        filename_map = {}
    for entry in sorted(set(items), key=str):
        if entry.path is None:
            continue
        path = output_dir or os.path.abspath(entry.path)
        ensuredir(path)
        try:
            name, obj, parent, modname = import_by_name(entry.name)
            qualname = name.replace(modname + '.', '')
        except ImportExceptionGroup as exc:
            try:
                name, obj, parent, modname = import_ivar_by_name(entry.name)
                qualname = name.replace(modname + '.', '')
            except ImportError as exc2:
                if exc2.__cause__:
                    exceptions: List[BaseException] = exc.exceptions + [exc2.__cause__]
                else:
                    exceptions = exc.exceptions + [exc2]
                errors = list({'* %s: %s' % (type(e).__name__, e) for e in exceptions})
                logger.warning(__('[autosummary] failed to import %s.\nPossible hints:\n%s'), entry.name, '\n'.join(errors))
                continue
        context: Dict[str, Any] = {}
        if app:
            context.update(app.config.autosummary_context)
        content = generate_autosummary_content(name, obj, parent, template, entry.template, imported_members, app, entry.recursive, context, modname, qualname)
        filename = os.path.join(path, filename_map.get(name, name) + suffix)
        if os.path.isfile(filename):
            with open(filename, encoding=encoding) as f:
                old_content = f.read()
            if content == old_content:
                continue
            elif overwrite:
                with open(filename, 'w', encoding=encoding) as f:
                    f.write(content)
                new_files.append(filename)
        else:
            with open(filename, 'w', encoding=encoding) as f:
                f.write(content)
            new_files.append(filename)
    if new_files:
        generate_autosummary_docs(new_files, output_dir=output_dir, suffix=suffix, base_path=base_path, imported_members=imported_members, app=app, overwrite=overwrite)