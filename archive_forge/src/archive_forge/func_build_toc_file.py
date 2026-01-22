from __future__ import annotations
import html
import os
from os import path
from typing import Any
from docutils import nodes
from docutils.nodes import Element, Node, document
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.config import Config
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import make_filename_from_project, relpath
from sphinx.util.template import SphinxRenderer
@progress_message(__('writing TOC file'))
def build_toc_file(self) -> None:
    """Create a ToC file (.hhp) on outdir."""
    filename = path.join(self.outdir, self.config.htmlhelp_basename + '.hhc')
    with open(filename, 'w', encoding=self.encoding, errors='xmlcharrefreplace') as f:
        toctree = self.env.get_and_resolve_doctree(self.config.master_doc, self, prune_toctrees=False)
        visitor = ToCTreeVisitor(toctree)
        matcher = NodeMatcher(addnodes.compact_paragraph, toctree=True)
        for node in toctree.traverse(matcher):
            node.walkabout(visitor)
        context = {'body': visitor.astext(), 'suffix': self.out_suffix, 'short_title': self.config.html_short_title, 'master_doc': self.config.master_doc, 'domain_indices': self.domain_indices}
        f.write(self.render('project.hhc', context))