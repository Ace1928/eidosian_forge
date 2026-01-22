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
def endErrorElementHandler(self, tag):
    """Handle end of an XML error element."""
    element = self.element
    if element is not None:
        self.parser.StartElementHandler = self.startElementHandler
        self.parser.EndElementHandler = self.endElementHandler
        self.parser.CharacterDataHandler = self.skipCharacterDataHandler
    data = ''.join(self.data)
    if data == '':
        return
    if self.ignore_errors is False:
        raise RuntimeError(data)
    self.data = []
    value = ErrorElement(data, tag)
    if element is None:
        self.record = element
    else:
        element.store(value)