from xml.sax._exceptions import *
from xml.sax.handler import feature_validation, feature_namespaces
from xml.sax.handler import feature_namespace_prefixes
from xml.sax.handler import feature_external_ges, feature_external_pes
from xml.sax.handler import feature_string_interning
from xml.sax.handler import property_xml_string, property_interning_dict
import sys
from xml.sax import xmlreader, saxutils, handler
def getProperty(self, name):
    if name == handler.property_lexical_handler:
        return self._lex_handler_prop
    elif name == property_interning_dict:
        return self._interning
    elif name == property_xml_string:
        if self._parser:
            if hasattr(self._parser, 'GetInputContext'):
                return self._parser.GetInputContext()
            else:
                raise SAXNotRecognizedException('This version of expat does not support getting the XML string')
        else:
            raise SAXNotSupportedException('XML string cannot be returned when not parsing')
    raise SAXNotRecognizedException("Property '%s' not recognized" % name)