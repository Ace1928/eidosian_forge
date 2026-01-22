import datetime
import importlib
from copy import copy
from types import ModuleType
from typing import TYPE_CHECKING, cast, Dict, Any, List, Iterator, \
from .exceptions import ElementPathTypeError
from .tdop import Token
from .namespaces import NamespacesType
from .datatypes import AnyAtomicType, Timezone, Language
from .protocols import ElementProtocol, DocumentProtocol
from .etree import is_etree_element, is_etree_document
from .xpath_nodes import ChildNodeType, XPathNode, AttributeNode, NamespaceNode, \
from .tree_builders import RootArgType, get_node_tree
def iter_followings(self) -> Iterator[ChildNodeType]:
    """Iterator for 'following' forward axis."""
    if isinstance(self.item, ElementNode):
        status = (self.item, self.axis)
        self.axis = 'following'
        descendants = set(self.item.iter_descendants())
        position = self.item.position
        root = self.item
        while isinstance(root.parent, ElementNode) and root is not self.root:
            root = root.parent
        for item in root.iter_descendants(with_self=False):
            if position < item.position and item not in descendants:
                self.item = item
                yield cast(ChildNodeType, self.item)
        self.item, self.axis = status