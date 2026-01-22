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
def startRawElementHandler(self, name, attrs):
    """Handle start of an XML raw element."""
    prefix = None
    if self.namespace_prefix:
        try:
            uri, name = name.split()
        except ValueError:
            pass
        else:
            prefix = self.namespace_prefix[uri]
            if self.namespace_level[prefix] == 1:
                attrs = {'xmlns': uri}
    if prefix:
        key = f'{prefix}:{name}'
    else:
        key = name
    tag = '<%s' % name
    for key, value in attrs.items():
        tag += f' {key}="{value}"'
    tag += '>'
    self.data.append(tag)
    self.parser.EndElementHandler = self.endRawElementHandler
    self.level += 1