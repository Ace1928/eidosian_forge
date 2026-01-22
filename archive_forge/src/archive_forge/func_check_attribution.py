import sys
import re
from types import FunctionType, MethodType
from docutils import nodes, statemachine, utils
from docutils import ApplicationError, DataError
from docutils.statemachine import StateMachineWS, StateWS
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.nodes import whitespace_normalize_name
import docutils.parsers.rst
from docutils.parsers.rst import directives, languages, tableparser, roles
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils import escape2null, unescape, column_width
from docutils.utils import punctuation_chars, roman, urischemes
from docutils.utils import split_escaped_whitespace
def check_attribution(self, indented, attribution_start):
    """
        Check attribution shape.
        Return the index past the end of the attribution, and the indent.
        """
    indent = None
    i = attribution_start + 1
    for i in range(attribution_start + 1, len(indented)):
        line = indented[i].rstrip()
        if not line:
            break
        if indent is None:
            indent = len(line) - len(line.lstrip())
        elif len(line) - len(line.lstrip()) != indent:
            return (None, None)
    else:
        i += 1
    return (i, indent or 0)