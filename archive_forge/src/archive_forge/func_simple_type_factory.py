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
def simple_type_factory(self, elem: ElementType, schema: Optional[SchemaType]=None, parent: Optional[XsdComponent]=None) -> XsdSimpleType:
    """
        Factory function for XSD simple types. Parses the xs:simpleType element and its
        child component, that can be a restriction, a list or a union. Annotations are
        linked to simple type instance, omitting the inner annotation if both are given.
        """
    if schema is None:
        schema = self
    annotation = None
    try:
        child = elem[0]
    except IndexError:
        return cast(XsdSimpleType, self.maps.types[XSD_ANY_SIMPLE_TYPE])
    else:
        if child.tag == XSD_ANNOTATION:
            annotation = XsdAnnotation(child, schema, parent)
            try:
                child = elem[1]
            except IndexError:
                msg = _('(restriction | list | union) expected')
                schema.parse_error(msg, elem)
                return cast(XsdSimpleType, self.maps.types[XSD_ANY_SIMPLE_TYPE])
    xsd_type: XsdSimpleType
    if child.tag == XSD_RESTRICTION:
        xsd_type = self.xsd_atomic_restriction_class(child, schema, parent)
    elif child.tag == XSD_LIST:
        xsd_type = self.xsd_list_class(child, schema, parent)
    elif child.tag == XSD_UNION:
        xsd_type = self.xsd_union_class(child, schema, parent)
    else:
        msg = _('(restriction | list | union) expected')
        schema.parse_error(msg, elem)
        return cast(XsdSimpleType, self.maps.types[XSD_ANY_SIMPLE_TYPE])
    if annotation is not None:
        xsd_type._annotation = annotation
    try:
        xsd_type.name = get_qname(schema.target_namespace, elem.attrib['name'])
    except KeyError:
        if parent is None:
            msg = _("missing attribute 'name' in a global simpleType")
            schema.parse_error(msg, elem)
            xsd_type.name = 'nameless_%s' % str(id(xsd_type))
    else:
        if parent is not None:
            msg = _("attribute 'name' not allowed for a local simpleType")
            schema.parse_error(msg, elem)
            xsd_type.name = None
    if 'final' in elem.attrib:
        try:
            xsd_type._final = get_xsd_derivation_attribute(elem, 'final')
        except ValueError as err:
            xsd_type.parse_error(err, elem)
    return xsd_type