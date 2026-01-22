import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def removeAttributeNode(self, node):
    if node is None:
        raise xml.dom.NotFoundErr()
    try:
        self._attrs[node.name]
    except KeyError:
        raise xml.dom.NotFoundErr()
    _clear_id_cache(self)
    node.unlink()
    node.ownerDocument = self.ownerDocument
    return node