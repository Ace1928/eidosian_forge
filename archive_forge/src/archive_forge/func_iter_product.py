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
def iter_product(self, selectors: Sequence[Callable[[Any], Any]], varnames: Optional[Sequence[str]]=None) -> Iterator[Any]:
    """
        Iterator for cartesian products of selectors.

        :param selectors: a sequence of selector generator functions.
        :param varnames: a sequence of variables for storing the generated values.
        """
    iterators = [x(self) for x in selectors]
    dimension = len(iterators)
    prod = [None] * dimension
    max_index = dimension - 1
    k = 0
    while True:
        try:
            value = next(iterators[k])
        except StopIteration:
            if not k:
                return
            iterators[k] = selectors[k](self)
            k -= 1
        else:
            if varnames is not None:
                try:
                    self.variables[varnames[k]] = value
                except (TypeError, IndexError):
                    pass
            prod[k] = value
            if k == max_index:
                yield tuple(prod)
            else:
                k += 1