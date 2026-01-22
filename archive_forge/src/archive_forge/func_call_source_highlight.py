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
def call_source_highlight(input, format):
    proc = subprocess.Popen(['source-highlight', '--out-format=html', '--no-doc', '--css=none', '--src-lang=%s' % format], shell=False, stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate(input)
    result = stdout
    proc.wait()
    return result