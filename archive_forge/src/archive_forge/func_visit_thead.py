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
def visit_thead(self, node):
    el = self.append_child('table:table-header-rows')
    self.set_current_element(el)
    self.in_thead = True
    self.paragraph_style_stack.append('Table_20_Heading')