import re
import codecs
import sys
from docutils import nodes
from docutils.utils import split_escaped_whitespace, escape2null, unescape
from docutils.parsers.rst.languages import en as _fallback_language_module
def value_or(values, other):
    """
    The argument can be any of `values` or `argument_type`.
    """

    def auto_or_other(argument):
        if argument in values:
            return argument
        else:
            return other(argument)
    return auto_or_other