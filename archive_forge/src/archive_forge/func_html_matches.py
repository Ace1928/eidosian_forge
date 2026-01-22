import subprocess
import doctest
import os
import sys
import shutil
import re
import cgi
import rfc822
from io import StringIO
from paste.util import PySourceColor
def html_matches(pattern, text):
    regex = re.escape(pattern)
    regex = regex.replace('\\.\\.\\.', '.*')
    regex = re.sub('0x[0-9a-f]+', '.*', regex)
    regex = '^%s$' % regex
    return re.search(regex, text)