from xml.sax._exceptions import *
from xml.sax.handler import feature_validation, feature_namespaces
from xml.sax.handler import feature_namespace_prefixes
from xml.sax.handler import feature_external_ges, feature_external_pes
from xml.sax.handler import feature_string_interning
from xml.sax.handler import property_xml_string, property_interning_dict
import sys
from xml.sax import xmlreader, saxutils, handler
def end_element_ns(self, name):
    pair = name.split()
    if len(pair) == 1:
        pair = (None, name)
    elif len(pair) == 3:
        pair = (pair[0], pair[1])
    else:
        pair = tuple(pair)
    self._cont_handler.endElementNS(pair, None)