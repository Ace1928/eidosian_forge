import re
from collections import Counter
from decimal import Decimal
from typing import Any, Callable, Iterator, List, MutableMapping, \
from xml.etree.ElementTree import ParseError
from .exceptions import XMLSchemaValueError, XMLSchemaTypeError
from .names import XSI_SCHEMA_LOCATION, XSI_NONS_SCHEMA_LOCATION
from .aliases import ElementType, NamespacesType, AtomicValueType, NumericValueType
def get_extended_qname(qname: str, namespaces: Optional[MutableMapping[str, str]]) -> str:
    """
    Get the extended form of a QName, using a namespace map.
    Local names are mapped to the default namespace.

    :param qname: a prefixed QName or a local name or an extended QName.
    :param namespaces: an optional mapping from prefixes to namespace URIs.
    """
    if not namespaces:
        return qname
    try:
        if qname[0] == '{':
            return qname
    except IndexError:
        return qname
    try:
        prefix, name = qname.split(':', 1)
    except ValueError:
        if not namespaces.get(''):
            return qname
        else:
            return f'{{{namespaces['']}}}{qname}'
    else:
        try:
            uri = namespaces[prefix]
        except KeyError:
            return qname
        else:
            return f'{{{uri}}}{name}' if uri else name