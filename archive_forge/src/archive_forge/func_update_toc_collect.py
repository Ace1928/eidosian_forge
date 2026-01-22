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
def update_toc_collect(self, el, level, collection):
    collection.append((level, el))
    level += 1
    for child_el in el.getchildren():
        if child_el.tag != 'text:index-body':
            self.update_toc_collect(child_el, level, collection)