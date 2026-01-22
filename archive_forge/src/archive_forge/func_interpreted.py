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
def interpreted(self, rawsource, text, role, lineno):
    role_fn, messages = roles.role(role, self.language, lineno, self.reporter)
    if role_fn:
        nodes, messages2 = role_fn(role, rawsource, text, lineno, self)
        try:
            nodes[0][0].rawsource = unescape(text, True)
        except IndexError:
            pass
        return (nodes, messages + messages2)
    else:
        msg = self.reporter.error('Unknown interpreted text role "%s".' % role, line=lineno)
        return ([self.problematic(rawsource, rawsource, msg)], messages + [msg])