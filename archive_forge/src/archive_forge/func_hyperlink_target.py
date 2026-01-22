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
def hyperlink_target(self, match):
    pattern = self.explicit.patterns.target
    lineno = self.state_machine.abs_line_number()
    block, indent, offset, blank_finish = self.state_machine.get_first_known_indented(match.end(), until_blank=True, strip_indent=False)
    blocktext = match.string[:match.end()] + '\n'.join(block)
    block = [escape2null(line) for line in block]
    escaped = block[0]
    blockindex = 0
    while True:
        targetmatch = pattern.match(escaped)
        if targetmatch:
            break
        blockindex += 1
        try:
            escaped += block[blockindex]
        except IndexError:
            raise MarkupError('malformed hyperlink target.')
    del block[:blockindex]
    block[0] = (block[0] + ' ')[targetmatch.end() - len(escaped) - 1:].strip()
    target = self.make_target(block, blocktext, lineno, targetmatch.group('name'))
    return ([target], blank_finish)