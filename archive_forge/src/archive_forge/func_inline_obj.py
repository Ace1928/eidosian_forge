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
def inline_obj(self, match, lineno, end_pattern, nodeclass, restore_backslashes=False):
    string = match.string
    matchstart = match.start('start')
    matchend = match.end('start')
    if self.quoted_start(match):
        return (string[:matchend], [], string[matchend:], [], '')
    endmatch = end_pattern.search(string[matchend:])
    if endmatch and endmatch.start(1):
        _text = endmatch.string[:endmatch.start(1)]
        text = unescape(_text, restore_backslashes)
        textend = matchend + endmatch.end(1)
        rawsource = unescape(string[matchstart:textend], True)
        node = nodeclass(rawsource, text)
        node[0].rawsource = unescape(_text, True)
        return (string[:matchstart], [node], string[textend:], [], endmatch.group(1))
    msg = self.reporter.warning('Inline %s start-string without end-string.' % nodeclass.__name__, line=lineno)
    text = unescape(string[matchstart:matchend], True)
    rawsource = unescape(string[matchstart:matchend], True)
    prb = self.problematic(text, rawsource, msg)
    return (string[:matchstart], [prb], string[matchend:], [msg], '')