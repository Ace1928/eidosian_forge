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
def iter_descendants(self, axis: Optional[str]=None) -> Iterator[Union[None, XPathNode]]:
    """
        Iterator for 'descendant' and 'descendant-or-self' forward axes and '//' shortcut.

        :param axis: the context axis, for default has no explicit axis.
        """
    if isinstance(self.item, (DocumentNode, ElementNode)):
        status = (self.item, self.axis)
        self.axis = axis
        for self.item in self.item.iter_descendants(with_self=axis != 'descendant'):
            yield self.item
        self.item, self.axis = status
    elif axis != 'descendant' and isinstance(self.item, XPathNode):
        self.axis, axis = (axis, self.axis)
        yield self.item
        self.axis = axis