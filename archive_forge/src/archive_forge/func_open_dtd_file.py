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
def open_dtd_file(self, filename):
    """Open specified DTD file."""
    if DataHandler.local_dtd_dir is not None:
        path = os.path.join(DataHandler.local_dtd_dir, filename)
        try:
            handle = open(path, 'rb')
        except FileNotFoundError:
            pass
        else:
            return handle
    path = os.path.join(DataHandler.global_dtd_dir, filename)
    try:
        handle = open(path, 'rb')
    except FileNotFoundError:
        pass
    else:
        return handle
    return None