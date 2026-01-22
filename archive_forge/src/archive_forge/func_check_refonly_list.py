from typing import Any, Dict, List, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.transforms import SphinxTransform
def check_refonly_list(node: Node) -> bool:
    """Check for list with only references in it."""
    visitor = RefOnlyListChecker(self.document)
    try:
        node.walk(visitor)
    except nodes.NodeFound:
        return False
    else:
        return True