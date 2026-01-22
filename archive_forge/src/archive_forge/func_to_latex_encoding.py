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
def to_latex_encoding(self, docutils_encoding):
    """Translate docutils encoding name into LaTeX's.

        Default method is remove "-" and "_" chars from docutils_encoding.
        """
    tr = {'iso-8859-1': 'latin1', 'iso-8859-2': 'latin2', 'iso-8859-3': 'latin3', 'iso-8859-4': 'latin4', 'iso-8859-5': 'iso88595', 'iso-8859-9': 'latin5', 'iso-8859-15': 'latin9', 'mac_cyrillic': 'maccyr', 'windows-1251': 'cp1251', 'koi8-r': 'koi8-r', 'koi8-u': 'koi8-u', 'windows-1250': 'cp1250', 'windows-1252': 'cp1252', 'us-ascii': 'ascii'}
    encoding = docutils_encoding.lower()
    if encoding in tr:
        return tr[encoding]
    encoding = encoding.replace('_', '').replace('-', '')
    return encoding.split(':')[0]