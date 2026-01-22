import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def get_page_width(self):
    """Return the document's page width in centimeters."""
    root = self.get_dom_stylesheet()
    nodes = root.iterfind('.//{urn:oasis:names:tc:opendocument:xmlns:style:1.0}page-layout/{urn:oasis:names:tc:opendocument:xmlns:style:1.0}page-layout-properties')
    width = None
    for node in nodes:
        page_width = node.get('{urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0}page-width')
        margin_left = node.get('{urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0}margin-left')
        margin_right = node.get('{urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0}margin-right')
        if page_width is None or margin_left is None or margin_right is None:
            continue
        try:
            page_width, _ = self.convert_to_cm(page_width)
            margin_left, _ = self.convert_to_cm(margin_left)
            margin_right, _ = self.convert_to_cm(margin_right)
        except ValueError:
            self.document.reporter.warning('Stylesheet file contains invalid page width or margin size.')
        width = page_width - margin_left - margin_right
    if width is None:
        width = 15.24
    return width