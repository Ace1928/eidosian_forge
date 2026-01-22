import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def read_partial(self, s, position=0, reentrances=None, fstruct=None):
    """
        Helper function that reads in a feature structure.

        :param s: The string to read.
        :param position: The position in the string to start parsing.
        :param reentrances: A dictionary from reentrance ids to values.
            Defaults to an empty dictionary.
        :return: A tuple (val, pos) of the feature structure created by
            parsing and the position where the parsed feature structure ends.
        :rtype: bool
        """
    if reentrances is None:
        reentrances = {}
    try:
        return self._read_partial(s, position, reentrances, fstruct)
    except ValueError as e:
        if len(e.args) != 2:
            raise
        self._error(s, *e.args)