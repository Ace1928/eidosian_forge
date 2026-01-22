import inspect
import re
import sys
import textwrap
from pprint import pformat
from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
def read_valuation(s, encoding=None):
    """
    Convert a valuation string into a valuation.

    :param s: a valuation string
    :type s: str
    :param encoding: the encoding of the input string, if it is binary
    :type encoding: str
    :return: a ``nltk.sem`` valuation
    :rtype: Valuation
    """
    if encoding is not None:
        s = s.decode(encoding)
    statements = []
    for linenum, line in enumerate(s.splitlines()):
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        try:
            statements.append(_read_valuation_line(line))
        except ValueError as e:
            raise ValueError(f'Unable to parse line {linenum}: {line}') from e
    return Valuation(statements)