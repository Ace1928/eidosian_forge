from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
def startPrefixMapping(self, prefix, uri):
    self._new_mappings[prefix] = uri
    try:
        self._ns_mapping[prefix].append(uri)
    except KeyError:
        self._ns_mapping[prefix] = [uri]
    if prefix is None:
        self._default_ns = uri