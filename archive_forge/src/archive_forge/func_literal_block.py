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
def literal_block(self):
    """Return a list of nodes."""
    indented, indent, offset, blank_finish = self.state_machine.get_indented()
    while indented and (not indented[-1].strip()):
        indented.trim_end()
    if not indented:
        return self.quoted_literal_block()
    data = '\n'.join(indented)
    literal_block = nodes.literal_block(data, data)
    literal_block.source, literal_block.line = self.state_machine.get_source_and_line(offset + 1)
    nodelist = [literal_block]
    if not blank_finish:
        nodelist.append(self.unindent_warning('Literal block'))
    return nodelist