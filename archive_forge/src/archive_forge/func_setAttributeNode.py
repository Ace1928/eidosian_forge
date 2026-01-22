import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def setAttributeNode(self, attr):
    if attr.ownerElement not in (None, self):
        raise xml.dom.InuseAttributeErr('attribute node already owned')
    self._ensure_attributes()
    old1 = self._attrs.get(attr.name, None)
    if old1 is not None:
        self.removeAttributeNode(old1)
    old2 = self._attrsNS.get((attr.namespaceURI, attr.localName), None)
    if old2 is not None and old2 is not old1:
        self.removeAttributeNode(old2)
    _set_attribute_node(self, attr)
    if old1 is not attr:
        return old1
    if old2 is not attr:
        return old2