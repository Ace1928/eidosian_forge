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
def xsd_qname(self, local_name: str) -> str:
    """Returns a prefixed QName string for XSD namespace."""
    if self.namespaces.get('xs') == XSD_NAMESPACE:
        return 'xs:%s' % local_name
    for pfx, uri in self.namespaces.items():
        if uri == XSD_NAMESPACE:
            return '%s:%s' % (pfx, local_name) if pfx else local_name
    raise xpath_error('XPST0081', 'Missing XSD namespace registration')