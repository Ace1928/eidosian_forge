from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extractall_same_as_extract_subject_index(any_string_dtype):
    mi = MultiIndex.from_tuples([('A', 'first'), ('B', 'second'), ('C', 'third')], names=('capital', 'ordinal'))
    s = Series(['a3', 'b3', 'c2'], index=mi, name='series_name', dtype=any_string_dtype)
    pattern_two_noname = '([a-z])([0-9])'
    extract_two_noname = s.str.extract(pattern_two_noname, expand=True)
    has_match_index = s.str.extractall(pattern_two_noname)
    no_match_index = has_match_index.xs(0, level='match')
    tm.assert_frame_equal(extract_two_noname, no_match_index)
    pattern_two_named = '(?P<letter>[a-z])(?P<digit>[0-9])'
    extract_two_named = s.str.extract(pattern_two_named, expand=True)
    has_match_index = s.str.extractall(pattern_two_named)
    no_match_index = has_match_index.xs(0, level='match')
    tm.assert_frame_equal(extract_two_named, no_match_index)
    pattern_one_named = '(?P<group_name>[a-z])'
    extract_one_named = s.str.extract(pattern_one_named, expand=True)
    has_match_index = s.str.extractall(pattern_one_named)
    no_match_index = has_match_index.xs(0, level='match')
    tm.assert_frame_equal(extract_one_named, no_match_index)
    pattern_one_noname = '([a-z])'
    extract_one_noname = s.str.extract(pattern_one_noname, expand=True)
    has_match_index = s.str.extractall(pattern_one_noname)
    no_match_index = has_match_index.xs(0, level='match')
    tm.assert_frame_equal(extract_one_noname, no_match_index)