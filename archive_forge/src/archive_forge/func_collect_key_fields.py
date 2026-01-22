import warnings
from copy import copy as _copy
from decimal import Decimal
from types import GeneratorType
from typing import TYPE_CHECKING, cast, Any, Dict, Iterator, List, Optional, \
from xml.etree.ElementTree import Element
from elementpath import XPath2Parser, ElementPathError, XPathContext, XPathToken, \
from elementpath.datatypes import AbstractDateTime, Duration, AbstractBinary
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_COMPLEX_TYPE, XSD_SIMPLE_TYPE, XSD_ALTERNATIVE, \
from ..aliases import ElementType, SchemaType, BaseXsdType, SchemaElementType, \
from ..translation import gettext as _
from ..helpers import get_qname, get_namespace, etree_iter_location_hints, \
from .. import dataobjects
from ..converters import ElementData, XMLSchemaConverter
from ..xpath import XsdSchemaProtocol, XsdElementProtocol, XMLSchemaProxy, \
from ..resources import XMLResource
from .exceptions import XMLSchemaNotBuiltError, XMLSchemaValidationError, \
from .helpers import get_xsd_derivation_attribute
from .xsdbase import XSD_TYPE_DERIVATIONS, XSD_ELEMENT_DERIVATIONS, \
from .particles import ParticleMixin, OccursCalculator
from .identities import XsdIdentity, XsdKey, XsdUnique, \
from .simple_types import XsdSimpleType
from .attributes import XsdAttribute
from .wildcards import XsdAnyElement
def collect_key_fields(self, elem: ElementType, xsd_type: BaseXsdType, validation: str='lax', nilled: bool=False, **kwargs: Any) -> Iterator[XMLSchemaValidationError]:
    element_node: Union[ElementNode, LazyElementNode]
    xsd_fields: Optional[IdentityCounterType]
    try:
        identities = kwargs['identities']
        resource = cast(XMLResource, kwargs['source'])
    except KeyError:
        return
    try:
        namespaces = kwargs['namespaces']
    except KeyError:
        namespaces = None
    element_node = resource.get_xpath_node(elem)
    xsd_element = self if self.ref is None else self.ref
    if xsd_element.type is not xsd_type:
        xsd_element = _copy(xsd_element)
        xsd_element.type = xsd_type
    for identity in self.selected_by:
        try:
            counter = identities[identity]
        except KeyError:
            continue
        else:
            if not counter.enabled or not identity.elements:
                continue
        if counter.elements is None:
            root_node = resource.get_xpath_node(counter.elem)
            context = XPathContext(root_node)
            assert identity.selector is not None and identity.selector.token is not None
            counter.elements = set(identity.selector.token.select_results(context))
        if elem not in counter.elements:
            continue
        try:
            if xsd_element.type is self.type and xsd_element in identity.elements:
                xsd_fields = identity.elements[xsd_element]
                if xsd_fields is None:
                    xsd_fields = identity.get_fields(xsd_element.xpath_node)
                    identity.elements[xsd_element] = xsd_fields
            else:
                xsd_fields = identity.get_fields(xsd_element.xpath_node)
            if all((x is None for x in xsd_fields)):
                continue
            decoders = cast(Tuple[XsdAttribute, ...], xsd_fields)
            fields = identity.get_fields(element_node, namespaces, decoders=decoders)
        except (XMLSchemaValueError, XMLSchemaTypeError) as err:
            yield self.validation_error(validation, err, elem, **kwargs)
        else:
            if any((x is not None for x in fields)) or nilled:
                try:
                    counter.increase(fields)
                except ValueError as err:
                    yield self.validation_error(validation, err, elem, **kwargs)