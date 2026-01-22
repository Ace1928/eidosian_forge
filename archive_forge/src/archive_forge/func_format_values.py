import re
import codecs
import sys
from docutils import nodes
from docutils.utils import split_escaped_whitespace, escape2null, unescape
from docutils.parsers.rst.languages import en as _fallback_language_module
def format_values(values):
    return '%s, or "%s"' % (', '.join(['"%s"' % s for s in values[:-1]]), values[-1])