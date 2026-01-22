from __future__ import annotations
import logging  # isort:skip
from os.path import basename, join
from docutils import nodes
from docutils.parsers.rst.directives import unchanged
from sphinx.directives.code import CodeBlock, container_wrapper, dedent_lines
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import logging, parselinenos
from sphinx.util.nodes import set_source_info
from . import PARALLEL_SAFE
from .templates import (
from .util import get_sphinx_resources
def get_js_source(self):
    js_file = self.options.get('js_file', False)
    if js_file and self.content:
        raise SphinxError("bokehjs-content:: directive can't have both js_file and content")
    if js_file:
        log.debug(f'[bokehjs-content] handling external example in {self.env.docname!r}: {js_file}')
        path = js_file
        if not js_file.startswith('/'):
            path = join(self.env.app.srcdir, path)
        js_source = open(path).read()
    else:
        log.debug(f'[bokehjs-content] handling inline example in {self.env.docname!r}')
        js_source = '\n'.join(self.content)
    return js_source