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
def has_unbalanced_braces(self, string):
    """Test whether there are unmatched '{' or '}' characters."""
    level = 0
    for ch in string:
        if ch == '{':
            level += 1
        if ch == '}':
            level -= 1
        if level < 0:
            return True
    return level != 0