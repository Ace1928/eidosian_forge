from decimal import DecimalException
from typing import cast, Any, Callable, Dict, Iterator, List, \
from xml.etree import ElementTree
from ..aliases import ElementType, AtomicValueType, ComponentClassType, \
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE, XSD_SIMPLE_TYPE, XSD_PATTERN, \
from ..translation import gettext as _
from ..helpers import local_name
from .exceptions import XMLSchemaValidationError, XMLSchemaEncodeError, \
from .xsdbase import XsdComponent, XsdType, ValidationMixin
from .facets import XsdFacet, XsdWhiteSpaceFacet, XsdPatternFacets, \
@property
def sequence_type(self) -> str:
    if self.is_empty():
        return 'empty-sequence()'
    root_type = self.root_type
    if root_type.name is not None:
        sequence_type = f'xs:{root_type.local_name}'
    else:
        sequence_type = 'xs:untypedAtomic'
    if not self.is_list():
        return sequence_type
    elif self.is_emptiable():
        return f'{sequence_type}*'
    else:
        return f'{sequence_type}+'