import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Type
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst.states import RSTState
from docutils.statemachine import StringList
from docutils.utils import Reporter, assemble_option_dict
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import Documenter, Options
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.nodes import nested_parse_with_titles
def process_documenter_options(documenter: Type[Documenter], config: Config, options: Dict) -> Options:
    """Recognize options of Documenter from user input."""
    for name in AUTODOC_DEFAULT_OPTIONS:
        if name not in documenter.option_spec:
            continue
        else:
            negated = options.pop('no-' + name, True) is None
            if name in config.autodoc_default_options and (not negated):
                if name in options and isinstance(config.autodoc_default_options[name], str):
                    if name in AUTODOC_EXTENDABLE_OPTIONS:
                        if options[name] is not None and options[name].startswith('+'):
                            options[name] = ','.join([config.autodoc_default_options[name], options[name][1:]])
                else:
                    options[name] = config.autodoc_default_options[name]
            elif options.get(name) is not None:
                options[name] = options[name].lstrip('+')
    return Options(assemble_option_dict(options.items(), documenter.option_spec))