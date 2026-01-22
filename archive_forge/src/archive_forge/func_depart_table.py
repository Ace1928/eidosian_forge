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
def depart_table(self, node):
    attribkey = add_ns('style:width', nsdict=SNSD)
    attribval = '%.4fin' % (self.table_width,)
    el1 = self.current_table_style
    el2 = el1[0]
    el2.attrib[attribkey] = attribval
    self.set_to_parent()