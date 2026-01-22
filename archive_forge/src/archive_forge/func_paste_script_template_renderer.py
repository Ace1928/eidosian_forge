from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def paste_script_template_renderer(content, vars, filename=None):
    tmpl = Template(content, name=filename)
    return tmpl.substitute(vars)