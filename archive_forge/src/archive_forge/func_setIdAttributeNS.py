import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def setIdAttributeNS(self, namespaceURI, localName):
    idAttr = self.getAttributeNodeNS(namespaceURI, localName)
    self.setIdAttributeNode(idAttr)