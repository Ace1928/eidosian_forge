from typing import Any, Dict, List, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.transforms.references import Substitutions
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.latex.nodes import (captioned_literal_block, footnotemark, footnotetext,
from sphinx.domains.citation import CitationDomain
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.nodes import NodeMatcher
def get_footnote_by_reference(self, node: nodes.footnote_reference) -> nodes.footnote:
    docname = node['docname']
    for footnote in self.footnotes:
        if docname == footnote['docname'] and footnote['ids'][0] == node['refid']:
            return footnote
    raise ValueError('No footnote not found for given reference node %r' % node)