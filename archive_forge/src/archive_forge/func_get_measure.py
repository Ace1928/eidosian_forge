import re
import codecs
import sys
from docutils import nodes
from docutils.utils import split_escaped_whitespace, escape2null, unescape
from docutils.parsers.rst.languages import en as _fallback_language_module
def get_measure(argument, units):
    """
    Check for a positive argument of one of the units and return a
    normalized string of the form "<value><unit>" (without space in
    between).

    To be called from directive option conversion functions.
    """
    match = re.match('^([0-9.]+) *(%s)$' % '|'.join(units), argument)
    try:
        float(match.group(1))
    except (AttributeError, ValueError):
        raise ValueError('not a positive measure of one of the following units:\n%s' % ' '.join(['"%s"' % i for i in units]))
    return match.group(1) + match.group(2)