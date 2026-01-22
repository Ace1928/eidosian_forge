import os
import warnings
from collections import Counter
from xml.parsers import expat
from io import BytesIO
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from urllib.request import urlopen
from urllib.parse import urlparse
from Bio import StreamModeError
def schemaHandler(self, name, attrs):
    """Process the XML schema (before processing the element)."""
    key = '%s noNamespaceSchemaLocation' % self.schema_namespace
    schema = attrs[key]
    handle = self.open_xsd_file(os.path.basename(schema))
    if not handle:
        handle = urlopen(schema)
        text = handle.read()
        self.save_xsd_file(os.path.basename(schema), text)
        handle.close()
        self.parse_xsd(ET.fromstring(text))
    else:
        self.parse_xsd(ET.fromstring(handle.read()))
        handle.close()
    self.startElementHandler(name, attrs)
    self.parser.StartElementHandler = self.startElementHandler