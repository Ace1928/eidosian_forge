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
def visit_inline(self, node):
    styles = node.attributes.get('classes', ())
    if styles:
        el = self.current_element
        for inline_style in styles:
            el = SubElement(el, 'text:span', attrib={'text:style-name': self.rststyle(inline_style)})
        count = len(styles)
    else:
        el = SubElement(self.current_element, 'text:span')
        count = 1
    self.set_current_element(el)
    self.inline_style_count_stack.append(count)