from abc import ABCMeta
import os
import logging
import threading
import warnings
import re
import sys
from copy import copy as _copy
from operator import attrgetter
from typing import cast, Callable, ItemsView, List, Optional, Dict, Any, \
from xml.etree.ElementTree import Element, ParseError
from elementpath import XPathToken, SchemaElementNode, build_schema_node_tree
from ..exceptions import XMLSchemaTypeError, XMLSchemaKeyError, XMLSchemaRuntimeError, \
from ..names import VC_MIN_VERSION, VC_MAX_VERSION, VC_TYPE_AVAILABLE, \
from ..aliases import ElementType, XMLSourceType, NamespacesType, LocationsType, \
from ..translation import gettext as _
from ..helpers import prune_etree, get_namespace, get_qname, is_defuse_error
from ..namespaces import NamespaceResourcesMap, NamespaceView
from ..resources import is_local_url, is_remote_url, url_path_is_file, \
from ..converters import XMLSchemaConverter
from ..xpath import XsdSchemaProtocol, XMLSchemaProxy, ElementPathMixin
from .. import dataobjects
from .exceptions import XMLSchemaParseError, XMLSchemaValidationError, XMLSchemaEncodeError, \
from .helpers import get_xsd_derivation_attribute
from .xsdbase import check_validation_mode, XsdValidator, XsdComponent, XsdAnnotation
from .notations import XsdNotation
from .identities import XsdIdentity, XsdKey, XsdKeyref, XsdUnique, \
from .facets import XSD_10_FACETS, XSD_11_FACETS
from .simple_types import XsdSimpleType, XsdList, XsdUnion, XsdAtomicRestriction, \
from .attributes import XsdAttribute, XsdAttributeGroup, Xsd11Attribute
from .complex_types import XsdComplexType, Xsd11ComplexType
from .groups import XsdGroup, Xsd11Group
from .elements import XsdElement, Xsd11Element
from .wildcards import XsdAnyElement, XsdAnyAttribute, Xsd11AnyElement, \
from .global_maps import XsdGlobals
def raw_decoder(self, source: XMLResource, path: Optional[str]=None, schema_path: Optional[str]=None, validation: str='lax', namespaces: Optional[NamespacesType]=None, **kwargs: Any) -> Iterator[Union[Any, XMLSchemaValidationError]]:
    """Returns a generator for decoding a resource."""
    if path:
        selector = source.iterfind(path, namespaces, nsmap=namespaces)
    else:
        selector = source.iter_depth(nsmap=namespaces)
    for elem in selector:
        xsd_element = self.get_element(elem.tag, schema_path, namespaces)
        if xsd_element is None:
            if XSI_TYPE in elem.attrib:
                xsd_element = self.create_element(name=elem.tag)
            else:
                reason = _('{!r} is not an element of the schema').format(elem)
                yield self.validation_error(validation, reason, elem, source, namespaces)
                continue
        yield from xsd_element.iter_decode(elem, validation, **kwargs)
    if 'max_depth' not in kwargs:
        yield from self._validate_references(source, validation=validation, **kwargs)