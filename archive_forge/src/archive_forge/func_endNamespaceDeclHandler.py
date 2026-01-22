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
def endNamespaceDeclHandler(self, prefix):
    """Handle end of an XML namespace declaration."""
    if prefix != 'xsi':
        self.namespace_level[prefix] -= 1
        if self.namespace_level[prefix] == 0:
            for key, value in self.namespace_prefix.items():
                if value == prefix:
                    break
            else:
                raise RuntimeError('Failed to find namespace prefix')
            del self.namespace_prefix[key]