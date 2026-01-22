import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def importNode(self, node, deep):
    if node.nodeType == Node.DOCUMENT_NODE:
        raise xml.dom.NotSupportedErr('cannot import document nodes')
    elif node.nodeType == Node.DOCUMENT_TYPE_NODE:
        raise xml.dom.NotSupportedErr('cannot import document type nodes')
    return _clone_node(node, deep, self)