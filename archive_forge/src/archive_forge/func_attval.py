import sys
import os
import time
import re
import string
import urllib.request, urllib.parse, urllib.error
from docutils import frontend, nodes, languages, writers, utils, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import pick_math_environment, unichar2tex
def attval(self, text, whitespace=re.compile('[\n\r\t\x0b\x0c]')):
    """Cleanse, encode, and return attribute value text."""
    return self.encode(whitespace.sub(' ', text))