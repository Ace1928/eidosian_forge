from collections import namedtuple
from collections.abc import MutableMapping, MutableSequence
from typing import TYPE_CHECKING, cast, Any, Dict, Iterator, Iterable, \
from xml.etree.ElementTree import Element
from ..exceptions import XMLSchemaTypeError
from ..names import XSI_NAMESPACE
from ..aliases import NamespacesType, BaseXsdType
from ..namespaces import NamespaceMapper
def map_content(self, content: Iterable[Tuple[str, Any, Any]]) -> Iterator[Tuple[str, Any, Any]]:
    """
        A generator function for converting decoded content to a data structure.
        If the instance has a not-empty map of namespaces registers the mapped URIs
        and prefixes.

        :param content: A sequence or an iterator of tuples with the name of the         element, the decoded value and the `XsdElement` instance associated.
        """
    if not content:
        return
    for name, value, xsd_child in content:
        try:
            if name[0] == '{':
                yield (self.map_qname(name), value, xsd_child)
            else:
                yield (name, value, xsd_child)
        except TypeError:
            if self.cdata_prefix is not None:
                yield (f'{self.cdata_prefix}{name}', value, xsd_child)