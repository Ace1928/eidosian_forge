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
def block_quote(self, indented, line_offset):
    elements = []
    while indented:
        blockquote_lines, attribution_lines, attribution_offset, indented, new_line_offset = self.split_attribution(indented, line_offset)
        blockquote = nodes.block_quote()
        self.nested_parse(blockquote_lines, line_offset, blockquote)
        elements.append(blockquote)
        if attribution_lines:
            attribution, messages = self.parse_attribution(attribution_lines, attribution_offset)
            blockquote += attribution
            elements += messages
        line_offset = new_line_offset
        while indented and (not indented[0]):
            indented = indented[1:]
            line_offset += 1
    return elements