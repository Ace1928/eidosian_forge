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
def is_namespace_allowed(self, namespace: str) -> bool:
    if self.not_namespace:
        return namespace not in self.not_namespace
    elif '##any' in self.namespace or namespace == XSI_NAMESPACE:
        return True
    elif '##other' in self.namespace:
        if not namespace:
            return False
        return namespace != self.target_namespace
    else:
        return namespace in self.namespace