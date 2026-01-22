from typing import cast, Any, Iterator, List, Optional, Union
from .namespaces import NamespacesType
from .exceptions import ElementPathTypeError
from .protocols import ElementProtocol, LxmlElementProtocol, \
from .etree import is_etree_document, is_etree_element
from .xpath_nodes import SchemaElemType, ChildNodeType, ElementMapType, \
def get_node_tree(root: RootArgType, namespaces: Optional[NamespacesType]=None, uri: Optional[str]=None, fragment: bool=False) -> Union[DocumentNode, ElementNode]:
    """
    Returns a tree of XPath nodes that wrap the provided root tree.

    :param root: an Element or an ElementTree or a schema or a schema element.
    :param namespaces: an optional mapping from prefixes to namespace URIs,     Ignored if root is a lxml etree or a schema structure.
    :param uri: an optional URI associated with the root element or the document.
    :param fragment: if `True` a root element is considered a fragment, otherwise     a root element is considered the root of an XML document. If the root is a     document node or an ElementTree instance, and fragment is `True` then use the     root element and returns an element node.
    """
    if isinstance(root, (DocumentNode, ElementNode)):
        if uri is not None and root.uri is None:
            root.uri = uri
        if fragment and isinstance(root, DocumentNode):
            root_node = root.getroot()
            if root_node.uri is None:
                root_node.uri = root.uri
            return root_node
        return root
    if is_etree_document(root):
        _root = cast(DocumentProtocol, root).getroot() if fragment else root
    elif is_etree_element(root) and (not callable(cast(ElementProtocol, root).tag)):
        _root = root
    else:
        msg = 'invalid root {!r}, an Element or an ElementTree or a schema node required'
        raise ElementPathTypeError(msg.format(root))
    if hasattr(_root, 'xpath'):
        return build_lxml_node_tree(cast(LxmlRootType, _root), uri, fragment)
    elif hasattr(_root, 'xsd_version') and hasattr(_root, 'maps'):
        return build_schema_node_tree(cast(SchemaElemType, root), uri)
    else:
        return build_node_tree(cast(ElementTreeRootType, root), namespaces, uri)