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
def get_docname_for_node(self, node: Node) -> str:
    while node:
        if isinstance(node, nodes.document):
            return self.env.path2doc(node['source']) or ''
        elif isinstance(node, addnodes.start_of_file):
            return node['docname']
        else:
            node = node.parent
    try:
        source = node['source']
    except TypeError:
        raise ValueError('Failed to get a docname!') from None
    raise ValueError(f'Failed to get a docname for source {source!r}!')