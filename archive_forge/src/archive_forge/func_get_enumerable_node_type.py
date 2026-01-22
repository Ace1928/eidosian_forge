import copy
from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List, NamedTuple, Optional,
from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst.states import Inliner
from sphinx.addnodes import pending_xref
from sphinx.errors import SphinxError
from sphinx.locale import _
from sphinx.roles import XRefRole
from sphinx.util.typing import RoleFunction
def get_enumerable_node_type(self, node: Node) -> Optional[str]:
    """Get type of enumerable nodes (experimental)."""
    enum_node_type, _ = self.enumerable_nodes.get(node.__class__, (None, None))
    return enum_node_type