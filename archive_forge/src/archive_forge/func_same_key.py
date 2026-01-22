import math
from decimal import Decimal
from functools import cmp_to_key
from itertools import zip_longest
from typing import Any, Callable, Optional, Iterable, Iterator
from .protocols import ElementProtocol
from .exceptions import xpath_error
from .datatypes import UntypedAtomic, AnyURI, AbstractQName
from .collations import UNICODE_CODEPOINT_COLLATION, CollationManager
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, NamespaceNode, \
from .xpath_tokens import XPathToken, XPathFunction, XPathMap, XPathArray
def same_key(k1: Any, k2: Any) -> bool:
    if isinstance(k1, (str, AnyURI, UntypedAtomic)):
        if not isinstance(k2, (str, AnyURI, UntypedAtomic)):
            return False
        return str(k1) == str(k2)
    elif isinstance(k1, float) and math.isnan(k1):
        return isinstance(k2, float) and math.isnan(k2)
    elif isinstance(k1, AbstractQName) ^ isinstance(k2, AbstractQName):
        return False
    try:
        return True if k1 == k2 else False
    except TypeError:
        return False