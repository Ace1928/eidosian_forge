from abc import ABCMeta
import locale
from collections.abc import MutableSequence
from urllib.parse import urlparse
from typing import cast, Any, Callable, ClassVar, Dict, List, \
from ..helpers import upper_camel_case, is_ncname, ordinal
from ..exceptions import ElementPathError, ElementPathTypeError, \
from ..namespaces import NamespacesType, XSD_NAMESPACE, XML_NAMESPACE, \
from ..collations import UNICODE_COLLATION_BASE_URI, UNICODE_CODEPOINT_COLLATION
from ..datatypes import UntypedAtomic, AtomicValueType, QName
from ..xpath_tokens import NargsType, XPathToken, ProxyToken, XPathFunction, XPathConstructor
from ..xpath_context import XPathContext, XPathSchemaContext
from ..sequence_types import is_sequence_type, match_sequence_type
from ..schema_proxy import AbstractSchemaProxy
from ..xpath1 import XPath1Parser
def is_schema_bound(self) -> bool:
    return self.schema is not None and 'symbol_table' in self.__dict__