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
@classmethod
def create_meta_schema(cls, source: Optional[str]=None, base_schemas: Union[None, Dict[str, str], List[Tuple[str, str]]]=None, global_maps: Optional[XsdGlobals]=None) -> SchemaType:
    """
        Creates a new meta-schema instance.

        :param source: an optional argument referencing to or containing the XSD meta-schema         resource. Required if the schema class doesn't already have a meta-schema.
        :param base_schemas: an optional dictionary that contains namespace URIs and         schema locations. If provided is used as substitute for class BASE_SCHEMAS.         Also a sequence of (namespace, location) items can be provided if there are more         schema documents for one or more namespaces.
        :param global_maps: is an optional argument containing an :class:`XsdGlobals`         instance for the new meta schema. If not provided a new map is created.
        """
    if source is None:
        if cls.meta_schema is None or not cls.meta_schema.url:
            raise XMLSchemaValueError(_('Missing meta-schema source URL'))
        source = cls.meta_schema.url
    _base_schemas: Union[ItemsView[str, str], List[Tuple[str, str]]]
    if base_schemas is None:
        _base_schemas = cls.BASE_SCHEMAS.items()
    elif isinstance(base_schemas, dict):
        _base_schemas = base_schemas.items()
    else:
        try:
            _base_schemas = [(n, l) for n, l in base_schemas]
        except ValueError:
            msg = _("The argument 'base_schemas' must be a dictionary or a sequence of couples")
            raise XMLSchemaValueError(msg) from None
    meta_schema: SchemaType
    meta_schema_class = cls if cls.meta_schema is None else cls.meta_schema.__class__
    if global_maps is None:
        meta_schema = meta_schema_class(source, XSD_NAMESPACE, defuse='never', build=False)
        global_maps = meta_schema.maps
    elif XSD_NAMESPACE not in global_maps.namespaces:
        meta_schema = meta_schema_class(source, XSD_NAMESPACE, global_maps=global_maps, defuse='never', build=False)
    else:
        meta_schema = global_maps.namespaces[XSD_NAMESPACE][0]
    for ns, location in _base_schemas:
        if ns == XSD_NAMESPACE:
            meta_schema.include_schema(location=location)
        elif ns not in global_maps.namespaces:
            meta_schema.import_schema(namespace=ns, location=location)
    return meta_schema