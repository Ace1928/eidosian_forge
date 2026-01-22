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
def fill_func2(self, matchobj):
    spaces = matchobj.group(0)
    repl = ' <text:s text:c="%d"/>' % (len(spaces) - 1,)
    return repl