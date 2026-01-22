from collections import deque
import os
import typing
from typing import (
from abc import ABC, abstractmethod
from enum import Enum
import string
import copy
import warnings
import re
import sys
from collections.abc import Iterable
import traceback
import types
from operator import itemgetter
from functools import wraps
from threading import RLock
from pathlib import Path
from .util import (
from .exceptions import *
from .actions import *
from .results import ParseResults, _ParseResultsWithOffset
from .unicode import pyparsing_unicode
def set_parse_action(self, *fns: ParseAction, **kwargs) -> 'ParserElement':
    """
        Define one or more actions to perform when successfully matching parse element definition.

        Parse actions can be called to perform data conversions, do extra validation,
        update external data structures, or enhance or replace the parsed tokens.
        Each parse action ``fn`` is a callable method with 0-3 arguments, called as
        ``fn(s, loc, toks)`` , ``fn(loc, toks)`` , ``fn(toks)`` , or just ``fn()`` , where:

        - ``s``    = the original string being parsed (see note below)
        - ``loc``  = the location of the matching substring
        - ``toks`` = a list of the matched tokens, packaged as a :class:`ParseResults` object

        The parsed tokens are passed to the parse action as ParseResults. They can be
        modified in place using list-style append, extend, and pop operations to update
        the parsed list elements; and with dictionary-style item set and del operations
        to add, update, or remove any named results. If the tokens are modified in place,
        it is not necessary to return them with a return statement.

        Parse actions can also completely replace the given tokens, with another ``ParseResults``
        object, or with some entirely different object (common for parse actions that perform data
        conversions). A convenient way to build a new parse result is to define the values
        using a dict, and then create the return value using :class:`ParseResults.from_dict`.

        If None is passed as the ``fn`` parse action, all previously added parse actions for this
        expression are cleared.

        Optional keyword arguments:

        - ``call_during_try`` = (default= ``False``) indicate if parse action should be run during
          lookaheads and alternate testing. For parse actions that have side effects, it is
          important to only call the parse action once it is determined that it is being
          called as part of a successful parse. For parse actions that perform additional
          validation, then call_during_try should be passed as True, so that the validation
          code is included in the preliminary "try" parses.

        Note: the default parsing behavior is to expand tabs in the input string
        before starting the parsing process.  See :class:`parse_string` for more
        information on parsing strings containing ``<TAB>`` s, and suggested
        methods to maintain a consistent view of the parsed string, the parse
        location, and line and column positions within the parsed string.

        Example::

            # parse dates in the form YYYY/MM/DD

            # use parse action to convert toks from str to int at parse time
            def convert_to_int(toks):
                return int(toks[0])

            # use a parse action to verify that the date is a valid date
            def is_valid_date(instring, loc, toks):
                from datetime import date
                year, month, day = toks[::2]
                try:
                    date(year, month, day)
                except ValueError:
                    raise ParseException(instring, loc, "invalid date given")

            integer = Word(nums)
            date_str = integer + '/' + integer + '/' + integer

            # add parse actions
            integer.set_parse_action(convert_to_int)
            date_str.set_parse_action(is_valid_date)

            # note that integer fields are now ints, not strings
            date_str.run_tests('''
                # successful parse - note that integer fields were converted to ints
                1999/12/31

                # fail - invalid date
                1999/13/31
                ''')
        """
    if list(fns) == [None]:
        self.parseAction = []
    else:
        if not all((callable(fn) for fn in fns)):
            raise TypeError('parse actions must be callable')
        self.parseAction = [_trim_arity(fn) for fn in fns]
        self.callDuringTry = kwargs.get('call_during_try', kwargs.get('callDuringTry', False))
    return self