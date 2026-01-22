from os import path
from typing import Any, Dict, List, Optional, Tuple, Union
from docutils import nodes
from docutils.nodes import Node
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment.adapters.toctree import TocTree
from sphinx.locale import __
from sphinx.util import logging, progress_message
from sphinx.util.console import darkgreen  # type: ignore
from sphinx.util.nodes import inline_all_toctrees
def fix_refuris(self, tree: Node) -> None:
    fname = self.config.root_doc + self.out_suffix
    for refnode in tree.findall(nodes.reference):
        if 'refuri' not in refnode:
            continue
        refuri = refnode['refuri']
        hashindex = refuri.find('#')
        if hashindex < 0:
            continue
        hashindex = refuri.find('#', hashindex + 1)
        if hashindex >= 0:
            refnode['refuri'] = fname + refuri[hashindex:]