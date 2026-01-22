from typing import cast, Any, Iterator, List, Optional, Union
from .namespaces import NamespacesType
from .exceptions import ElementPathTypeError
from .protocols import ElementProtocol, LxmlElementProtocol, \
from .etree import is_etree_document, is_etree_element
from .xpath_nodes import SchemaElemType, ChildNodeType, ElementMapType, \
def build_document_node() -> ElementNode:
    nonlocal position
    nonlocal child
    for e in reversed([x for x in elem.itersiblings(preceding=True)]):
        if e.tag.__name__ == 'Comment':
            parent.children.append(CommentNode(e, parent, position))
        else:
            parent.children.append(ProcessingInstructionNode(e, parent, position))
        position += 1
    node = build_lxml_element_node()
    parent.children.append(node)
    for e in elem.itersiblings():
        if e.tag.__name__ == 'Comment':
            parent.children.append(CommentNode(e, parent, position))
        else:
            parent.children.append(ProcessingInstructionNode(e, parent, position))
        position += 1
    return node