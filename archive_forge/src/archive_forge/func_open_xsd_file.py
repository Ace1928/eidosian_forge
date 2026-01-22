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
def open_xsd_file(self, filename):
    """Open specified XSD file."""
    if DataHandler.local_xsd_dir is not None:
        path = os.path.join(DataHandler.local_xsd_dir, filename)
        try:
            handle = open(path, 'rb')
        except FileNotFoundError:
            pass
        else:
            return handle
    path = os.path.join(DataHandler.global_xsd_dir, filename)
    try:
        handle = open(path, 'rb')
    except FileNotFoundError:
        pass
    else:
        return handle
    return None