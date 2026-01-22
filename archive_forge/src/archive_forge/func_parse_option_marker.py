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
def parse_option_marker(self, match):
    """
        Return a list of `node.option` and `node.option_argument` objects,
        parsed from an option marker match.

        :Exception: `MarkupError` for invalid option markers.
        """
    optlist = []
    optionstrings = match.group().rstrip().split(', ')
    for optionstring in optionstrings:
        tokens = optionstring.split()
        delimiter = ' '
        firstopt = tokens[0].split('=', 1)
        if len(firstopt) > 1:
            tokens[:1] = firstopt
            delimiter = '='
        elif len(tokens[0]) > 2 and (tokens[0].startswith('-') and (not tokens[0].startswith('--')) or tokens[0].startswith('+')):
            tokens[:1] = [tokens[0][:2], tokens[0][2:]]
            delimiter = ''
        if len(tokens) > 1 and (tokens[1].startswith('<') and tokens[-1].endswith('>')):
            tokens[1:] = [' '.join(tokens[1:])]
        if 0 < len(tokens) <= 2:
            option = nodes.option(optionstring)
            option += nodes.option_string(tokens[0], tokens[0])
            if len(tokens) > 1:
                option += nodes.option_argument(tokens[1], tokens[1], delimiter=delimiter)
            optlist.append(option)
        else:
            raise MarkupError('wrong number of option tokens (=%s), should be 1 or 2: "%s"' % (len(tokens), optionstring))
    return optlist