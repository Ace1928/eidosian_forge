import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def read_logic_value(self, s, position, reentrances, match):
    try:
        try:
            expr = self._logic_parser.parse(match.group(1))
        except LogicalExpressionException as e:
            raise ValueError from e
        return (expr, match.end())
    except ValueError as e:
        raise ValueError('logic expression', match.start(1)) from e