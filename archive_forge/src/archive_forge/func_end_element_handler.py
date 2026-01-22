from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def end_element_handler(self, name):
    curNode = self.curNode
    if ' ' in name:
        uri, localname, prefix, qname = _parse_ns_name(self, name)
        assert curNode.namespaceURI == uri and curNode.localName == localname and (curNode.prefix == prefix), 'element stack messed up! (namespace)'
    else:
        assert curNode.nodeName == name, 'element stack messed up - bad nodeName'
        assert curNode.namespaceURI == EMPTY_NAMESPACE, 'element stack messed up - bad namespaceURI'
    self.curNode = curNode.parentNode
    self._finish_end_element(curNode)