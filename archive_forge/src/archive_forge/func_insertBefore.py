import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def insertBefore(self, newChild, refChild):
    raise xml.dom.HierarchyRequestErr('cannot insert children below an entity node')