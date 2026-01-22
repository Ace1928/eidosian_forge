from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def notation_decl_handler(self, notationName, base, systemId, publicId):
    node = self.document._create_notation(notationName, publicId, systemId)
    self.document.doctype.notations._seq.append(node)
    if self._filter and self._filter.acceptNode(node) == FILTER_ACCEPT:
        del self.document.doctype.notations._seq[-1]