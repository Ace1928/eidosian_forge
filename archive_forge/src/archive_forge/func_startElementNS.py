from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
def startElementNS(self, ns_name, qname, attributes=None):
    el_name = self._buildTag(ns_name)
    if attributes:
        attrs = {}
        try:
            iter_attributes = attributes.iteritems()
        except AttributeError:
            iter_attributes = attributes.items()
        for name_tuple, value in iter_attributes:
            if name_tuple[0]:
                attr_name = '{%s}%s' % name_tuple
            else:
                attr_name = name_tuple[1]
            attrs[attr_name] = value
    else:
        attrs = None
    element_stack = self._element_stack
    if self._root is None:
        element = self._root = self._makeelement(el_name, attrs, self._new_mappings)
        if self._root_siblings and hasattr(element, 'addprevious'):
            for sibling in self._root_siblings:
                element.addprevious(sibling)
        del self._root_siblings[:]
    else:
        element = SubElement(element_stack[-1], el_name, attrs, self._new_mappings)
    element_stack.append(element)
    self._new_mappings.clear()