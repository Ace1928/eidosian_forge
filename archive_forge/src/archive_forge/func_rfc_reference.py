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
def rfc_reference(self, match, lineno):
    text = match.group(0)
    if text.startswith('RFC'):
        rfcnum = int(match.group('rfcnum'))
        ref = self.document.settings.rfc_base_url + self.rfc_url % rfcnum
    else:
        raise MarkupMismatch
    unescaped = unescape(text)
    return [nodes.reference(unescape(text, True), unescaped, refuri=ref)]