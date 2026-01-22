from xml.sax._exceptions import *
from xml.sax.handler import feature_validation, feature_namespaces
from xml.sax.handler import feature_namespace_prefixes
from xml.sax.handler import feature_external_ges, feature_external_pes
from xml.sax.handler import feature_string_interning
from xml.sax.handler import property_xml_string, property_interning_dict
import sys
from xml.sax import xmlreader, saxutils, handler
def external_entity_ref(self, context, base, sysid, pubid):
    if not self._external_ges:
        return 1
    source = self._ent_handler.resolveEntity(pubid, sysid)
    source = saxutils.prepare_input_source(source, self._source.getSystemId() or '')
    self._entity_stack.append((self._parser, self._source))
    self._parser = self._parser.ExternalEntityParserCreate(context)
    self._source = source
    try:
        xmlreader.IncrementalParser.parse(self, source)
    except:
        return 0
    self._parser, self._source = self._entity_stack[-1]
    del self._entity_stack[-1]
    return 1