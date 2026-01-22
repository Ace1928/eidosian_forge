from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def getParser(self):
    """Return the parser object, creating a new one if needed."""
    if not self._parser:
        self._parser = self.createParser()
        self._intern_setdefault = self._parser.intern.setdefault
        self._parser.buffer_text = True
        self._parser.ordered_attributes = True
        self._parser.specified_attributes = True
        self.install(self._parser)
    return self._parser