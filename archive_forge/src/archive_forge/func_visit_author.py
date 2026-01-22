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
def visit_author(self, node):
    if isinstance(node.parent, nodes.authors):
        el = self.append_p('blockindent')
    else:
        el = self.generate_labeled_block(node, 'author')
    self.set_current_element(el)