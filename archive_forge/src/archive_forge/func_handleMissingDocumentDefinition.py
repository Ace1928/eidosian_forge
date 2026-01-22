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
def handleMissingDocumentDefinition(self, tag, attrs):
    """Raise an Exception if neither a DTD nor an XML Schema is found."""
    raise ValueError('As the XML data contained neither a Document Type Definition (DTD) nor an XML Schema, Bio.Entrez is unable to parse these data. We recommend using a generic XML parser from the Python standard library instead, for example ElementTree.')