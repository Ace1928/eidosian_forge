from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def attlist_decl_handler(self, elem, name, type, default, required):
    info = self._elem_info.get(elem)
    if info is None:
        info = ElementInfo(elem)
        self._elem_info[elem] = info
    info._attr_info.append([None, name, None, None, default, 0, type, required])