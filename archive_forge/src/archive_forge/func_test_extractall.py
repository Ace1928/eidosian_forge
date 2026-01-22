from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extractall(any_string_dtype):
    data = ['dave@google.com', 'tdhock5@gmail.com', 'maudelaperriere@gmail.com', 'rob@gmail.com some text steve@gmail.com', 'a@b.com some text c@d.com and e@f.com', np.nan, '']
    expected_tuples = [('dave', 'google', 'com'), ('tdhock5', 'gmail', 'com'), ('maudelaperriere', 'gmail', 'com'), ('rob', 'gmail', 'com'), ('steve', 'gmail', 'com'), ('a', 'b', 'com'), ('c', 'd', 'com'), ('e', 'f', 'com')]
    pat = '\n    (?P<user>[a-z0-9]+)\n    @\n    (?P<domain>[a-z]+)\n    \\.\n    (?P<tld>[a-z]{2,4})\n    '
    expected_columns = ['user', 'domain', 'tld']
    s = Series(data, dtype=any_string_dtype)
    expected_index = MultiIndex.from_tuples([(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2)], names=(None, 'match'))
    expected = DataFrame(expected_tuples, expected_index, expected_columns, dtype=any_string_dtype)
    result = s.str.extractall(pat, flags=re.VERBOSE)
    tm.assert_frame_equal(result, expected)
    mi = MultiIndex.from_tuples([('single', 'Dave'), ('single', 'Toby'), ('single', 'Maude'), ('multiple', 'robAndSteve'), ('multiple', 'abcdef'), ('none', 'missing'), ('none', 'empty')])
    s = Series(data, index=mi, dtype=any_string_dtype)
    expected_index = MultiIndex.from_tuples([('single', 'Dave', 0), ('single', 'Toby', 0), ('single', 'Maude', 0), ('multiple', 'robAndSteve', 0), ('multiple', 'robAndSteve', 1), ('multiple', 'abcdef', 0), ('multiple', 'abcdef', 1), ('multiple', 'abcdef', 2)], names=(None, None, 'match'))
    expected = DataFrame(expected_tuples, expected_index, expected_columns, dtype=any_string_dtype)
    result = s.str.extractall(pat, flags=re.VERBOSE)
    tm.assert_frame_equal(result, expected)
    s = Series(data, index=mi, dtype=any_string_dtype)
    s.index.names = ('matches', 'description')
    expected_index.names = ('matches', 'description', 'match')
    expected = DataFrame(expected_tuples, expected_index, expected_columns, dtype=any_string_dtype)
    result = s.str.extractall(pat, flags=re.VERBOSE)
    tm.assert_frame_equal(result, expected)