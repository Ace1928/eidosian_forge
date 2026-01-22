import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def replaceWholeText(self, content):
    parent = self.parentNode
    n = self.previousSibling
    while n is not None:
        if n.nodeType in (Node.TEXT_NODE, Node.CDATA_SECTION_NODE):
            next = n.previousSibling
            parent.removeChild(n)
            n = next
        else:
            break
    n = self.nextSibling
    if not content:
        parent.removeChild(self)
    while n is not None:
        if n.nodeType in (Node.TEXT_NODE, Node.CDATA_SECTION_NODE):
            next = n.nextSibling
            parent.removeChild(n)
            n = next
        else:
            break
    if content:
        self.data = content
        return self
    else:
        return None