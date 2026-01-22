from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def character_data_handler_cdata(self, data):
    childNodes = self.curNode.childNodes
    if self._cdata:
        if self._cdata_continue and childNodes[-1].nodeType == CDATA_SECTION_NODE:
            childNodes[-1].appendData(data)
            return
        node = self.document.createCDATASection(data)
        self._cdata_continue = True
    elif childNodes and childNodes[-1].nodeType == TEXT_NODE:
        node = childNodes[-1]
        value = node.data + data
        node.data = value
        return
    else:
        node = minidom.Text()
        node.data = data
        node.ownerDocument = self.document
    _append_child(self.curNode, node)