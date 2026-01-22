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
def unrestrict(self, node: Element) -> None:
    if self.restricted == node:
        self.restricted = None
        pos = node.parent.index(node)
        for i, footnote in enumerate(self.pendings):
            fntext = footnotetext('', *footnote.children, ids=footnote['ids'])
            node.parent.insert(pos + i + 1, fntext)
        self.pendings = []