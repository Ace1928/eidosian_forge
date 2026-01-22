import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def removeAttributeNS(self, namespaceURI, localName):
    if self._attrsNS is None:
        raise xml.dom.NotFoundErr()
    try:
        attr = self._attrsNS[namespaceURI, localName]
    except KeyError:
        raise xml.dom.NotFoundErr()
    self.removeAttributeNode(attr)