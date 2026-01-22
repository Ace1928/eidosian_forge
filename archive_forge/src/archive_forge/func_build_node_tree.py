from typing import cast, Any, Iterator, List, Optional, Union
from .namespaces import NamespacesType
from .exceptions import ElementPathTypeError
from .protocols import ElementProtocol, LxmlElementProtocol, \
from .etree import is_etree_document, is_etree_element
from .xpath_nodes import SchemaElemType, ChildNodeType, ElementMapType, \
def build_node_tree(root: ElementTreeRootType, namespaces: Optional[NamespacesType]=None, uri: Optional[str]=None) -> Union[DocumentNode, ElementNode]:
    """
    Returns a tree of XPath nodes that wrap the provided root tree.

    :param root: an Element or an ElementTree.
    :param namespaces: an optional mapping from prefixes to namespace URIs.
    :param uri: an optional URI associated with the document or the root element.
    """
    root_node: Union[DocumentNode, ElementNode]
    parent: Any
    elements: Any
    child: ChildNodeType
    children: Iterator[Any]
    position = 1

    def build_element_node() -> ElementNode:
        nonlocal position
        node = ElementNode(elem, parent, position, namespaces)
        position += 1
        elements[elem] = node
        position += len(node.nsmap) + int('xml' not in node.nsmap) + len(elem.attrib)
        if elem.text is not None:
            node.children.append(TextNode(elem.text, node, position))
            position += 1
        return node
    if hasattr(root, 'parse'):
        document = cast(DocumentProtocol, root)
        parent = DocumentNode(document, uri, position)
        root_node = parent
        position += 1
        elements = root_node.elements
        root_elem = document.getroot()
        if root_elem is None:
            return root_node
        elem = root_elem
        child = build_element_node()
        parent.children.append(child)
        parent = child
    else:
        elem = root
        parent = None
        elements = {}
        root_node = parent = build_element_node()
        root_node.elements = elements
        if uri is not None:
            root_node.uri = uri
    children = iter(elem)
    iterators: List[Any] = []
    ancestors: List[Any] = []
    while True:
        for elem in children:
            if not callable(elem.tag):
                child = build_element_node()
            elif elem.tag.__name__ == 'Comment':
                child = CommentNode(elem, parent, position)
                position += 1
            else:
                child = ProcessingInstructionNode(elem, parent, position)
            parent.children.append(child)
            if elem.tail is not None:
                parent.children.append(TextNode(elem.tail, parent, position))
                position += 1
            if len(elem):
                ancestors.append(parent)
                parent = child
                iterators.append(children)
                children = iter(elem)
                break
        else:
            try:
                children, parent = (iterators.pop(), ancestors.pop())
            except IndexError:
                return root_node