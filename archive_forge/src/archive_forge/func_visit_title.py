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
def visit_title(self, node, move_ids=1, title_type='title'):
    if isinstance(node.parent, docutils.nodes.section):
        section_level = self.section_level
        if section_level > 7:
            self.document.reporter.warning('Heading/section levels greater than 7 not supported.')
            self.document.reporter.warning('    Reducing to heading level 7 for heading: "%s"' % (node.astext(),))
            section_level = 7
        el1 = self.append_child('text:h', attrib={'text:outline-level': '%d' % section_level, 'text:style-name': self.rststyle('heading%d', (section_level,))})
        self.append_pending_ids(el1)
        self.set_current_element(el1)
    elif isinstance(node.parent, docutils.nodes.document):
        el1 = SubElement(self.current_element, 'text:p', attrib={'text:style-name': self.rststyle(title_type)})
        self.append_pending_ids(el1)
        text = node.astext()
        self.title = text
        self.found_doc_title = True
        self.set_current_element(el1)