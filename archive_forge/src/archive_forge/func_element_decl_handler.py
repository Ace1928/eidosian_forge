from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def element_decl_handler(self, name, model):
    info = self._elem_info.get(name)
    if info is None:
        self._elem_info[name] = ElementInfo(name, model)
    else:
        assert info._model is None
        info._model = model