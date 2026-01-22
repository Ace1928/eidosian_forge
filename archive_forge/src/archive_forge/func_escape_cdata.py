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
def escape_cdata(text):
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    ascii = ''
    for char in text:
        if ord(char) >= ord('\x7f'):
            ascii += '&#x%X;' % (ord(char),)
        else:
            ascii += char
    return ascii