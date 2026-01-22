from typing import cast, Any, Iterator, List, Optional, Union
from .namespaces import NamespacesType
from .exceptions import ElementPathTypeError
from .protocols import ElementProtocol, LxmlElementProtocol, \
from .etree import is_etree_document, is_etree_element
from .xpath_nodes import SchemaElemType, ChildNodeType, ElementMapType, \
def build_schema_node_tree(root: SchemaElemType, uri: Optional[str]=None, elements: Optional[ElementMapType]=None, global_elements: Optional[List[ChildNodeType]]=None) -> SchemaElementNode:
    """
    Returns a tree of XPath nodes that wrap the provided XSD schema structure.

    :param root: a schema or a schema element.
    :param uri: an optional URI associated with the root element.
    :param elements: a shared map from XSD elements to tree nodes. Provided for     linking together parts of the same schema or other schemas.
    :param global_elements: a list for schema global elements, used for linking     the elements declared by reference.
    """
    parent: Any
    elem: Any
    child: SchemaElementNode
    children: Iterator[Any]
    position = 1
    _elements = {} if elements is None else elements

    def build_schema_element_node() -> SchemaElementNode:
        nonlocal position
        node = SchemaElementNode(elem, parent, position, elem.namespaces)
        position += 1
        _elements[elem] = node
        position += len(elem.namespaces) + int('xml' not in elem.namespaces) + len(elem.attrib)
        return node
    children = iter(root)
    elem = root
    parent = None
    root_node = parent = build_schema_element_node()
    root_node.elements = _elements
    if uri is not None:
        root_node.uri = uri
    if global_elements is not None:
        global_elements.append(root_node)
    elif is_schema(root):
        global_elements = root_node.children
    else:
        global_elements = []
    local_nodes = {root: root_node}
    ref_nodes: List[SchemaElementNode] = []
    iterators: List[Any] = []
    ancestors: List[Any] = []
    while True:
        for elem in children:
            child = build_schema_element_node()
            child.xsd_type = elem.type
            parent.children.append(child)
            if elem in local_nodes:
                if elem.ref is None:
                    child.children = local_nodes[elem].children
                else:
                    ref_nodes.append(child)
            else:
                local_nodes[elem] = child
                if elem.ref is None:
                    ancestors.append(parent)
                    parent = child
                    iterators.append(children)
                    children = iter(elem)
                    break
                else:
                    ref_nodes.append(child)
        else:
            try:
                children, parent = (iterators.pop(), ancestors.pop())
            except IndexError:
                for element_node in ref_nodes:
                    elem = element_node.elem
                    ref = cast(XsdElementProtocol, elem.ref)
                    other: Any
                    for other in global_elements:
                        if other.elem is ref:
                            element_node.ref = other
                            break
                    else:
                        element_node.ref = build_schema_node_tree(ref, elements=_elements, global_elements=global_elements)
                return root_node