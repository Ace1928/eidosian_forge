from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element
from sphinx.addnodes import pending_xref
from sphinx.domains import Domain
from sphinx.locale import __
from sphinx.transforms import SphinxTransform
from sphinx.util import logging
from sphinx.util.nodes import copy_source_info, make_refnode
def note_citation(self, node: nodes.citation) -> None:
    label = node[0].astext()
    if label in self.citations:
        path = self.env.doc2path(self.citations[label][0])
        logger.warning(__('duplicate citation %s, other instance in %s'), label, path, location=node, type='ref', subtype='citation')
    self.citations[label] = (node['docname'], node['ids'][0], node.line)