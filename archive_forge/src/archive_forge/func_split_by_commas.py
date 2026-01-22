import collections.abc
import math
import re
import unicodedata
import urllib
from oslo_utils._i18n import _
from oslo_utils import encodeutils
def split_by_commas(value):
    """Split values by commas and quotes according to api-wg

    :param value: value to be split

    .. versionadded:: 3.17
    """
    import pyparsing as pp
    word = pp.QuotedString(quoteChar='"', escChar='\\') | pp.Word(pp.printables, excludeChars='",')
    grammar = pp.stringStart + pp.delimitedList(word) + pp.stringEnd
    try:
        return list(grammar.parseString(value))
    except pp.ParseException:
        raise ValueError('Invalid value: %s' % value)