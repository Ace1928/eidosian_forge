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
def parse_directive_block(self, indented, line_offset, directive, option_presets):
    option_spec = directive.option_spec
    has_content = directive.has_content
    if indented and (not indented[0].strip()):
        indented.trim_start()
        line_offset += 1
    while indented and (not indented[-1].strip()):
        indented.trim_end()
    if indented and (directive.required_arguments or directive.optional_arguments or option_spec):
        for i, line in enumerate(indented):
            if not line.strip():
                break
        else:
            i += 1
        arg_block = indented[:i]
        content = indented[i + 1:]
        content_offset = line_offset + i + 1
    else:
        content = indented
        content_offset = line_offset
        arg_block = []
    if option_spec:
        options, arg_block = self.parse_directive_options(option_presets, option_spec, arg_block)
    else:
        options = {}
    if arg_block and (not (directive.required_arguments or directive.optional_arguments)):
        content = arg_block + indented[i:]
        content_offset = line_offset
        arg_block = []
    while content and (not content[0].strip()):
        content.trim_start()
        content_offset += 1
    if directive.required_arguments or directive.optional_arguments:
        arguments = self.parse_directive_arguments(directive, arg_block)
    else:
        arguments = []
    if content and (not has_content):
        raise MarkupError('no content permitted')
    return (arguments, options, content, content_offset)