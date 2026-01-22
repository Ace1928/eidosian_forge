import re
from abc import ABCMeta
from copy import copy
from itertools import count
from typing import TYPE_CHECKING, cast, overload, Any, Dict, List, Iterator, \
from elementpath import XPathContext, XPath2Parser, build_node_tree, protocols
from elementpath.etree import etree_tostring
from .exceptions import XMLSchemaAttributeError, XMLSchemaTypeError, XMLSchemaValueError
from .aliases import ElementType, XMLSourceType, NamespacesType, BaseXsdType, DecodeType
from .helpers import get_namespace, get_prefixed_qname, local_name, raw_xml_encode
from .converters import ElementData, XMLSchemaConverter
from .resources import XMLResource
from . import validators
def fromsource(cls, source: Union[XMLSourceType, XMLResource], allow: str='all', defuse: str='remote', timeout: int=300, **kwargs: Any) -> DecodeType[Any]:
    if not isinstance(source, XMLResource):
        source = XMLResource(source, allow=allow, defuse=defuse, timeout=timeout)
    if 'converter' not in kwargs:
        kwargs['converter'] = DataBindingConverter
    return cls.xsd_element.schema.decode(source, **kwargs)