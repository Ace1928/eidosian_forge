import datetime
import email
from io import StringIO
from urllib.parse import urlparse
from django.utils.encoding import iri_to_uri
from django.utils.xmlutils import SimplerXMLGenerator
def write_items(self, handler):
    for item in self.items:
        handler.startElement('entry', self.item_attributes(item))
        self.add_item_elements(handler, item)
        handler.endElement('entry')