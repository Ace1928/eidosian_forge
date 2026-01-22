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
def visit_citation_reference(self, node):
    if self.settings.create_links:
        id = node.attributes['refid']
        el = self.append_child('text:reference-ref', attrib={'text:ref-name': '%s' % (id,), 'text:reference-format': 'text'})
        el.text = '['
        self.set_current_element(el)
    elif self.current_element.text is None:
        self.current_element.text = '['
    else:
        self.current_element.text += '['