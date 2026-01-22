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
def fill_line(self, line):
    line = FILL_PAT1.sub(self.fill_func1, line)
    line = FILL_PAT2.sub(self.fill_func2, line)
    return line