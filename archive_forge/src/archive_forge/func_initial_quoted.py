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
def initial_quoted(self, match, context, next_state):
    """Match arbitrary quote character on the first line only."""
    self.remove_transition('initial_quoted')
    quote = match.string[0]
    pattern = re.compile(re.escape(quote), re.UNICODE)
    self.add_transition('quoted', (pattern, self.quoted, self.__class__.__name__))
    self.initial_lineno = self.state_machine.abs_line_number()
    return ([match.string], next_state, [])