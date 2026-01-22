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
def parse_directive_arguments(self, directive, arg_block):
    required = directive.required_arguments
    optional = directive.optional_arguments
    arg_text = '\n'.join(arg_block)
    arguments = arg_text.split()
    if len(arguments) < required:
        raise MarkupError('%s argument(s) required, %s supplied' % (required, len(arguments)))
    elif len(arguments) > required + optional:
        if directive.final_argument_whitespace:
            arguments = arg_text.split(None, required + optional - 1)
        else:
            raise MarkupError('maximum %s argument(s) allowed, %s supplied' % (required + optional, len(arguments)))
    return arguments