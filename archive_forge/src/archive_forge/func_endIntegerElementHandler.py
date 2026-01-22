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
def endIntegerElementHandler(self, tag):
    """Handle end of an XML integer element."""
    attributes = self.attributes
    self.attributes = None
    assert tag == 'Item'
    key = attributes['Name']
    del attributes['Name']
    if self.data:
        value = int(''.join(self.data))
        self.data = []
        value = IntegerElement(value, tag, attributes, key)
    else:
        value = NoneElement(tag, attributes, key)
    element = self.element
    if element is None:
        self.record = value
    else:
        self.parser.EndElementHandler = self.endElementHandler
        self.parser.CharacterDataHandler = self.skipCharacterDataHandler
        if value is None:
            return
        element.store(value)