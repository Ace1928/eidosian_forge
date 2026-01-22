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
def visit_term(self, node):
    el = self.append_p('deflist-term-%d' % self.def_list_level)
    el.text = node.astext()
    self.set_current_element(el)
    raise nodes.SkipChildren()