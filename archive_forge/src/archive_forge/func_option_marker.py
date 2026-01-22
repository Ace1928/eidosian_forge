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
def option_marker(self, match, context, next_state):
    """Option list item."""
    try:
        option_list_item, blank_finish = self.option_list_item(match)
    except MarkupError:
        self.invalid_input()
    self.parent += option_list_item
    self.blank_finish = blank_finish
    return ([], next_state, [])