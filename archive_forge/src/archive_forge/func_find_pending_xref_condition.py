import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.domains import Domain
from sphinx.errors import NoUri
from sphinx.locale import __
from sphinx.transforms import SphinxTransform
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.nodes import find_pending_xref_condition, process_only_nodes
def find_pending_xref_condition(self, node: pending_xref, conditions: Sequence[str]) -> Optional[List[Node]]:
    for condition in conditions:
        matched = find_pending_xref_condition(node, condition)
        if matched:
            return matched.children
    else:
        return None