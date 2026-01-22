from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def write_unicode(file, text):
    if isinstance(text, unicode):
        text = text.encode(ENCODING, 'backslashreplace')
    file.write(text)