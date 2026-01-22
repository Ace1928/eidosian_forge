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
def xmlDeclHandler(self, version, encoding, standalone):
    """Set XML handlers when an XML declaration is found."""
    self.parser.CharacterDataHandler = self.characterDataHandler
    self.parser.ExternalEntityRefHandler = self.externalEntityRefHandler
    self.parser.StartNamespaceDeclHandler = self.startNamespaceDeclHandler
    self.parser.EndNamespaceDeclHandler = self.endNamespaceDeclHandler
    self.parser.StartElementHandler = self.handleMissingDocumentDefinition