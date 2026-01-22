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
def update_toc_add_numbers(self, collection):
    for level, el1 in collection:
        if el1.tag == 'text:p' and el1.text != 'Table of Contents':
            el2 = SubElement(el1, 'text:tab')
            el2.tail = '9999'