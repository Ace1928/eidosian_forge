from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def entity_decl_handler(self, entityName, is_parameter_entity, value, base, systemId, publicId, notationName):
    if is_parameter_entity:
        return
    if not self._options.entities:
        return
    node = self.document._create_entity(entityName, publicId, systemId, notationName)
    if value is not None:
        child = self.document.createTextNode(value)
        node.childNodes.append(child)
    self.document.doctype.entities._seq.append(node)
    if self._filter and self._filter.acceptNode(node) == FILTER_REJECT:
        del self.document.doctype.entities._seq[-1]