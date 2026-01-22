from __future__ import annotations
from sphinx.util import logging  # isort:skip
import re
import warnings
from os import getenv
from os.path import basename, dirname, join
from uuid import uuid4
from docutils import nodes
from docutils.parsers.rst.directives import choice, flag
from sphinx.errors import SphinxError
from sphinx.util import copyfile, ensuredir
from sphinx.util.display import status_iterator
from sphinx.util.nodes import set_source_info
from bokeh.document import Document
from bokeh.embed import autoload_static
from bokeh.model import Model
from bokeh.util.warnings import BokehDeprecationWarning
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .example_handler import ExampleHandler
from .util import _REPO_TOP, get_sphinx_resources
def process_args_or_content(self):
    if self.arguments and self.content:
        raise SphinxError("bokeh-plot:: directive can't have both args and content")
    if self.content:
        log.debug(f'[bokeh-plot] handling inline content in {self.env.docname!r}')
        path = self.env.bokeh_plot_auxdir
        return ('\n'.join(self.content), path)
    path = self.arguments[0]
    log.debug(f'[bokeh-plot] handling external content in {self.env.docname!r}: {path}')
    if path.startswith('__REPO__/'):
        path = join(_REPO_TOP, path.replace('__REPO__/', ''))
    elif not path.startswith('/'):
        path = join(self.env.app.srcdir, path)
    try:
        with open(path) as f:
            return (f.read(), path)
    except Exception as e:
        raise SphinxError(f'bokeh-plot:: error reading {path!r} for {self.env.docname!r}: {e!r}')