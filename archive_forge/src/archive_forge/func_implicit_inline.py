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
def implicit_inline(self, text, lineno):
    """
        Check each of the patterns in `self.implicit_dispatch` for a match,
        and dispatch to the stored method for the pattern.  Recursively check
        the text before and after the match.  Return a list of `nodes.Text`
        and inline element nodes.
        """
    if not text:
        return []
    for pattern, method in self.implicit_dispatch:
        match = pattern.search(text)
        if match:
            try:
                return self.implicit_inline(text[:match.start()], lineno) + method(match, lineno) + self.implicit_inline(text[match.end():], lineno)
            except MarkupMismatch:
                pass
    return [nodes.Text(unescape(text), rawsource=unescape(text, True))]