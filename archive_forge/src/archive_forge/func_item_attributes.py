import datetime
import email
from io import StringIO
from urllib.parse import urlparse
from django.utils.encoding import iri_to_uri
from django.utils.xmlutils import SimplerXMLGenerator
def item_attributes(self, item):
    """
        Return extra attributes to place on each item (i.e. item/entry) element.
        """
    return {}