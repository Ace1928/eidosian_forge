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
def parse_directive_options(self, option_presets, option_spec, arg_block):
    options = option_presets.copy()
    for i, line in enumerate(arg_block):
        if re.match(Body.patterns['field_marker'], line):
            opt_block = arg_block[i:]
            arg_block = arg_block[:i]
            break
    else:
        opt_block = []
    if opt_block:
        success, data = self.parse_extension_options(option_spec, opt_block)
        if success:
            options.update(data)
        else:
            raise MarkupError(data)
    return (options, arg_block)