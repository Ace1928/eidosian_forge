import re
from re import escape
from os.path import commonprefix
from itertools import groupby
from operator import itemgetter
Return a compiled regex that matches any string in the given list.

    The strings to match must be literal strings, not regexes.  They will be
    regex-escaped.

    *prefix* and *suffix* are pre- and appended to the final regex.
    