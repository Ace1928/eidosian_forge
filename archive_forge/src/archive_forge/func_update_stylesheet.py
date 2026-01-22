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
def update_stylesheet(self, stylesheet_root, language_code, region_code):
    """Update xml style sheet element with language and region/country."""
    updated = False
    modified_nodes = set()
    if language_code is not None or region_code is not None:
        n1 = stylesheet_root.find('{urn:oasis:names:tc:opendocument:xmlns:office:1.0}styles')
        if n1 is None:
            raise RuntimeError("Cannot find 'styles' element in styles.odt/styles.xml")
        n2_nodes = n1.findall('{urn:oasis:names:tc:opendocument:xmlns:style:1.0}default-style')
        if not n2_nodes:
            raise RuntimeError("Cannot find 'default-style' element in styles.xml")
        for node in n2_nodes:
            family = node.attrib.get('{urn:oasis:names:tc:opendocument:xmlns:style:1.0}family')
            if family == 'paragraph' or family == 'graphic':
                n3 = node.find('{urn:oasis:names:tc:opendocument:xmlns:style:1.0}text-properties')
                if n3 is None:
                    raise RuntimeError("Cannot find 'text-properties' element in styles.xml")
                if language_code is not None:
                    n3.attrib['{urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0}language'] = language_code
                    n3.attrib['{urn:oasis:names:tc:opendocument:xmlns:style:1.0}language-complex'] = language_code
                    updated = True
                    modified_nodes.add(n3)
                if region_code is not None:
                    n3.attrib['{urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0}country'] = region_code
                    n3.attrib['{urn:oasis:names:tc:opendocument:xmlns:style:1.0}country-complex'] = region_code
                    updated = True
                    modified_nodes.add(n3)
    return (updated, stylesheet_root, modified_nodes)