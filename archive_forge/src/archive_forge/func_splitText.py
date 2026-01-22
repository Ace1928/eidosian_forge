import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def splitText(self, offset):
    if offset < 0 or offset > len(self.data):
        raise xml.dom.IndexSizeErr('illegal offset value')
    newText = self.__class__()
    newText.data = self.data[offset:]
    newText.ownerDocument = self.ownerDocument
    next = self.nextSibling
    if self.parentNode and self in self.parentNode.childNodes:
        if next is None:
            self.parentNode.appendChild(newText)
        else:
            self.parentNode.insertBefore(newText, next)
    self.data = self.data[:offset]
    return newText