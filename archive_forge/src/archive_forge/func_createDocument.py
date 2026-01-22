import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def createDocument(self, namespaceURI, qualifiedName, doctype):
    if doctype and doctype.parentNode is not None:
        raise xml.dom.WrongDocumentErr('doctype object owned by another DOM tree')
    doc = self._create_document()
    add_root_element = not (namespaceURI is None and qualifiedName is None and (doctype is None))
    if not qualifiedName and add_root_element:
        raise xml.dom.InvalidCharacterErr('Element with no name')
    if add_root_element:
        prefix, localname = _nssplit(qualifiedName)
        if prefix == 'xml' and namespaceURI != 'http://www.w3.org/XML/1998/namespace':
            raise xml.dom.NamespaceErr("illegal use of 'xml' prefix")
        if prefix and (not namespaceURI):
            raise xml.dom.NamespaceErr('illegal use of prefix without namespaces')
        element = doc.createElementNS(namespaceURI, qualifiedName)
        if doctype:
            doc.appendChild(doctype)
        doc.appendChild(element)
    if doctype:
        doctype.parentNode = doctype.ownerDocument = doc
    doc.doctype = doctype
    doc.implementation = self
    return doc