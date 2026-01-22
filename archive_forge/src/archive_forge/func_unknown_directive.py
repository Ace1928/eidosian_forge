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
def unknown_directive(self, type_name):
    lineno = self.state_machine.abs_line_number()
    indented, indent, offset, blank_finish = self.state_machine.get_first_known_indented(0, strip_indent=False)
    text = '\n'.join(indented)
    error = self.reporter.error('Unknown directive type "%s".' % type_name, nodes.literal_block(text, text), line=lineno)
    return ([error], blank_finish)