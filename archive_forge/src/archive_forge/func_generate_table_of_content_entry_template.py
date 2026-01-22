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
def generate_table_of_content_entry_template(self, el1):
    for idx in range(1, 11):
        el2 = SubElement(el1, 'text:table-of-content-entry-template', attrib={'text:outline-level': '%d' % (idx,), 'text:style-name': self.rststyle('contents-%d' % (idx,))})
        SubElement(el2, 'text:index-entry-chapter')
        SubElement(el2, 'text:index-entry-text')
        SubElement(el2, 'text:index-entry-tab-stop', attrib={'style:leader-char': '.', 'style:type': 'right'})
        SubElement(el2, 'text:index-entry-page-number')