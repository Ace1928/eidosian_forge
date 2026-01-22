import re
from abc import ABCMeta
from typing import cast, Any, ClassVar, Dict, MutableMapping, \
from ..exceptions import MissingContextError, ElementPathValueError, \
from ..datatypes import QName
from ..tdop import Token, Parser
from ..namespaces import NamespacesType, XML_NAMESPACE, XSD_NAMESPACE, \
from ..sequence_types import match_sequence_type
from ..schema_proxy import AbstractSchemaProxy
from ..xpath_tokens import NargsType, XPathToken, XPathAxis, XPathFunction, \
@property
def other_namespaces(self) -> Dict[str, str]:
    """The subset of namespaces not known by default."""
    return {k: v for k, v in self.namespaces.items() if k not in self.DEFAULT_NAMESPACES or self.DEFAULT_NAMESPACES[k] != v}