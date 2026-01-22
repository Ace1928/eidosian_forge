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
def visit_docinfo(self, node):
    self.section_level += 1
    self.section_count += 1
    if self.settings.create_sections:
        el = self.append_child('text:section', attrib={'text:name': 'Section%d' % self.section_count, 'text:style-name': 'Sect%d' % self.section_level})
        self.set_current_element(el)