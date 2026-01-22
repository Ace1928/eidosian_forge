from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def pi_handler(self, target, data):
    node = self.document.createProcessingInstruction(target, data)
    _append_child(self.curNode, node)
    if self._filter and self._filter.acceptNode(node) == FILTER_REJECT:
        self.curNode.removeChild(node)