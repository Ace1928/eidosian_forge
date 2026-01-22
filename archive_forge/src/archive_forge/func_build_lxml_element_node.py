from typing import cast, Any, Iterator, List, Optional, Union
from .namespaces import NamespacesType
from .exceptions import ElementPathTypeError
from .protocols import ElementProtocol, LxmlElementProtocol, \
from .etree import is_etree_document, is_etree_element
from .xpath_nodes import SchemaElemType, ChildNodeType, ElementMapType, \
def build_lxml_element_node() -> ElementNode:
    nonlocal position
    node = ElementNode(elem, parent, position, elem.nsmap)
    position += 1
    elements[elem] = node
    position += len(elem.nsmap) + int('xml' not in elem.nsmap) + len(elem.attrib)
    if elem.text is not None:
        node.children.append(TextNode(elem.text, node, position))
        position += 1
    return node