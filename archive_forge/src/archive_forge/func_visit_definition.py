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
def visit_definition(self, node):
    self.paragraph_style_stack.append(self.rststyle('deflist-def-%d' % self.def_list_level))
    self.bumped_list_level_stack.append(ListLevel(1))