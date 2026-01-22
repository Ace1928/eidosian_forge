import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def getAttributeNodeNS(self, namespaceURI, localName):
    if self._attrsNS is None:
        return None
    return self._attrsNS.get((namespaceURI, localName))