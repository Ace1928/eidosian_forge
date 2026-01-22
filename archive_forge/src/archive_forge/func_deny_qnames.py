from typing import cast, Any, Callable, Dict, Iterable, Iterator, List, Optional, \
from elementpath import SchemaElementNode, build_schema_node_tree
from ..exceptions import XMLSchemaValueError
from ..names import XSI_NAMESPACE, XSD_ANY, XSD_ANY_ATTRIBUTE, \
from ..aliases import ElementType, SchemaType, SchemaElementType, SchemaAttributeType, \
from ..translation import gettext as _
from ..helpers import get_namespace, raw_xml_encode
from ..xpath import XsdSchemaProtocol, XsdElementProtocol, XMLSchemaProxy, ElementPathMixin
from .xsdbase import ValidationMixin, XsdComponent
from .particles import ParticleMixin
from . import elements
def deny_qnames(self, names: Iterable[str]) -> bool:
    if self.not_namespace:
        return all((x in self.not_qname or get_namespace(x) in self.not_namespace for x in names))
    elif '##any' in self.namespace:
        return all((x in self.not_qname for x in names))
    elif '##other' in self.namespace:
        return all((x in self.not_qname or get_namespace(x) == self.target_namespace for x in names))
    else:
        return all((x in self.not_qname or get_namespace(x) not in self.namespace for x in names))