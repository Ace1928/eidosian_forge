from __future__ import print_function, absolute_import
import threading
import warnings
from lxml import etree as _etree
from .common import DTDForbidden, EntitiesForbidden, NotSupportedError
def createDefaultParser(self):
    parser = _etree.XMLParser(**self.parser_config)
    element_class = self.element_class
    if self.element_class is not None:
        lookup = _etree.ElementDefaultClassLookup(element=element_class)
        parser.set_element_class_lookup(lookup)
    return parser