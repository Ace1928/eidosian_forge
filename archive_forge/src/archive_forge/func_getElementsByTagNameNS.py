import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def getElementsByTagNameNS(self, namespaceURI, localName):
    return _get_elements_by_tagName_ns_helper(self, namespaceURI, localName, NodeList())