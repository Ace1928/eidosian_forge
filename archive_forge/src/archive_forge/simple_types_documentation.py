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

        :param name: the XSD type's qualified name.
        :param python_type: the correspondent Python's type. If a tuple of types         is provided uses the first and consider the others as compatible types.
        :param base_type: the reference base type, None if it's a primitive type.
        :param admitted_facets: admitted facets tags for type (required for primitive types).
        :param facets: optional facets validators.
        :param to_python: optional decode function.
        :param from_python: optional encode function.
        