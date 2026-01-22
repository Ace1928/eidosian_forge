import re
import math
import operator
from abc import abstractmethod
from typing import TYPE_CHECKING, cast, Any, List, Optional, Pattern, Union, \
from xml.etree.ElementTree import Element
from elementpath import XPath2Parser, XPathContext, ElementPathError, \
from ..names import XSD_LENGTH, XSD_MIN_LENGTH, XSD_MAX_LENGTH, XSD_ENUMERATION, \
from ..aliases import ElementType, SchemaType, AtomicValueType, BaseXsdType
from ..translation import gettext as _
from ..helpers import count_digits, local_name
from .exceptions import XMLSchemaValidationError, XMLSchemaDecodeError
from .xsdbase import XsdComponent, XsdAnnotation
@property
def base_facet(self) -> Optional['XsdFacet']:
    """
        An object of the same type if the instance has a base facet, `None` otherwise.
        """
    base_type: Optional[BaseXsdType] = self.base_type
    tag = self.elem.tag
    while True:
        if base_type is None:
            return None
        try:
            base_facet = base_type.facets[tag]
        except (AttributeError, KeyError):
            base_type = base_type.base_type
        else:
            assert isinstance(base_facet, self.__class__)
            return base_facet