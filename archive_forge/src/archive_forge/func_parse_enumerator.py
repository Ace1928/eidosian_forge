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
def parse_enumerator(self, match, expected_sequence=None):
    """
        Analyze an enumerator and return the results.

        :Return:
            - the enumerator format ('period', 'parens', or 'rparen'),
            - the sequence used ('arabic', 'loweralpha', 'upperroman', etc.),
            - the text of the enumerator, stripped of formatting, and
            - the ordinal value of the enumerator ('a' -> 1, 'ii' -> 2, etc.;
              ``None`` is returned for invalid enumerator text).

        The enumerator format has already been determined by the regular
        expression match. If `expected_sequence` is given, that sequence is
        tried first. If not, we check for Roman numeral 1. This way,
        single-character Roman numerals (which are also alphabetical) can be
        matched. If no sequence has been matched, all sequences are checked in
        order.
        """
    groupdict = match.groupdict()
    sequence = ''
    for format in self.enum.formats:
        if groupdict[format]:
            break
    else:
        raise ParserError('enumerator format not matched')
    text = groupdict[format][self.enum.formatinfo[format].start:self.enum.formatinfo[format].end]
    if text == '#':
        sequence = '#'
    elif expected_sequence:
        try:
            if self.enum.sequenceregexps[expected_sequence].match(text):
                sequence = expected_sequence
        except KeyError:
            raise ParserError('unknown enumerator sequence: %s' % sequence)
    elif text == 'i':
        sequence = 'lowerroman'
    elif text == 'I':
        sequence = 'upperroman'
    if not sequence:
        for sequence in self.enum.sequences:
            if self.enum.sequenceregexps[sequence].match(text):
                break
        else:
            raise ParserError('enumerator sequence not matched')
    if sequence == '#':
        ordinal = 1
    else:
        try:
            ordinal = self.enum.converters[sequence](text)
        except roman.InvalidRomanNumeralError:
            ordinal = None
    return (format, sequence, text, ordinal)