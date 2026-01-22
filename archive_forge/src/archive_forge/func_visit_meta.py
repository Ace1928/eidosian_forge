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
def visit_meta(self, node):
    name = node.attributes.get('name')
    content = node.attributes.get('content')
    if name is not None and content is not None:
        self.meta_dict[name] = content