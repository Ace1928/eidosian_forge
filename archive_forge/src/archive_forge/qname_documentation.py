from typing import Any, Optional
from ..helpers import QNAME_PATTERN
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic

    XPath compliant QName, bound with a prefix and a namespace.

    :param uri: the bound namespace URI, must be a not empty     URI if a prefixed name is provided for the 2nd argument.
    :param qname: the prefixed name or a local name.
    