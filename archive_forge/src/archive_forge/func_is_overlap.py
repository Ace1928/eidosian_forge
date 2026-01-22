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
def is_overlap(self, other: ModelParticleType) -> bool:
    if not isinstance(other, XsdAnyElement):
        if isinstance(other, elements.XsdElement):
            return other.is_overlap(self)
        return False
    if self.not_namespace:
        if other.not_namespace:
            return True
        elif '##any' in other.namespace:
            return True
        elif '##other' in other.namespace:
            return True
        else:
            return any((ns not in self.not_namespace for ns in other.namespace))
    elif other.not_namespace:
        if '##any' in self.namespace:
            return True
        elif '##other' in self.namespace:
            return True
        else:
            return any((ns not in other.not_namespace for ns in self.namespace))
    elif self.namespace == other.namespace:
        return True
    elif '##any' in self.namespace or '##any' in other.namespace:
        return True
    elif '##other' in self.namespace:
        return any((ns and ns != self.target_namespace for ns in other.namespace))
    elif '##other' in other.namespace:
        return any((ns and ns != other.target_namespace for ns in self.namespace))
    else:
        return any((ns in self.namespace for ns in other.namespace))