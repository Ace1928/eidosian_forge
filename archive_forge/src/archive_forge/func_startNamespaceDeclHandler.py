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
def startNamespaceDeclHandler(self, prefix, uri):
    """Handle start of an XML namespace declaration."""
    if prefix == 'xsi':
        self.schema_namespace = uri
        self.parser.StartElementHandler = self.schemaHandler
    else:
        if prefix == 'mml':
            assert uri == 'http://www.w3.org/1998/Math/MathML'
        elif prefix == 'xlink':
            assert uri == 'http://www.w3.org/1999/xlink'
        elif prefix == 'ali':
            assert uri.rstrip('/') == 'http://www.niso.org/schemas/ali/1.0'
        else:
            raise ValueError(f"Unknown prefix '{prefix}' with uri '{uri}'")
        self.namespace_level[prefix] += 1
        self.namespace_prefix[uri] = prefix