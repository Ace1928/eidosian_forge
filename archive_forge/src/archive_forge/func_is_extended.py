from importlib import import_module
from urllib.parse import urljoin
from types import ModuleType
from typing import cast, Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union
from .datatypes import UntypedAtomic, get_atomic_value, AtomicValueType
from .namespaces import XML_NAMESPACE, XML_BASE, XSI_NIL, \
from .protocols import ElementProtocol, DocumentProtocol, XsdElementProtocol, \
from .helpers import match_wildcard, is_absolute_uri
from .etree import etree_iter_strings, is_etree_element, is_etree_document
def is_extended(self) -> bool:
    """
        Returns `True` if the document node cannot be represented with an
        ElementTree structure, `False` otherwise.
        """
    root = self.document.getroot()
    if root is None or not is_etree_element(root):
        return True
    elif not self.children:
        raise RuntimeError('Missing document root')
    elif len(self.children) == 1:
        return not isinstance(self.children[0], ElementNode)
    elif not hasattr(root, 'itersiblings'):
        return True
    elif any((isinstance(x, TextNode) for x in root)):
        return True
    else:
        return sum((isinstance(x, ElementNode) for x in root)) != 1