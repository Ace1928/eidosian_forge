import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def writexml(self, writer, indent='', addindent='', newl='', encoding=None, standalone=None):
    declarations = []
    if encoding:
        declarations.append(f'encoding="{encoding}"')
    if standalone is not None:
        declarations.append(f'standalone="{('yes' if standalone else 'no')}"')
    writer.write(f'<?xml version="1.0" {' '.join(declarations)}?>{newl}')
    for node in self.childNodes:
        node.writexml(writer, indent, addindent, newl)