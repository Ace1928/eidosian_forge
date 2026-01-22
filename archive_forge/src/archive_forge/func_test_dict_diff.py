import os
from shutil import rmtree
import pytest
from nipype.utils.misc import (
def test_dict_diff():
    abtuple = [('a', 'b')]
    abdict = dict(abtuple)
    assert dict_diff(abdict, abdict) == ''
    assert dict_diff(abdict, abtuple) == ''
    assert dict_diff(abtuple, abdict) == ''
    assert dict_diff(abtuple, abtuple) == ''
    diff = dict_diff({'a': 'b'}, {'b': 'a'})
    assert 'Dictionaries had differing keys' in diff
    assert "keys not previously seen: {'b'}" in diff
    assert "keys not presently seen: {'a'}" in diff
    complicated_val1 = [{'a': ['b'], 'c': ('d', 'e')}]
    complicated_val2 = [{'a': ['x'], 'c': ('d', 'e')}]
    uniformized_val1 = ({'a': ('b',), 'c': ('d', 'e')},)
    uniformized_val2 = ({'a': ('x',), 'c': ('d', 'e')},)
    diff = dict_diff({'a': complicated_val1}, {'a': complicated_val2})
    assert 'Some dictionary entries had differing values:' in diff
    assert 'a: {!r} != {!r}'.format(uniformized_val2, uniformized_val1) in diff
    diff = dict_diff({'a': 'b' * 60}, {'a': 'c' * 70})
    assert 'Some dictionary entries had differing values:' in diff
    assert "a: 'cccccccccc...cccccccccc' != 'bbbbbbbbbb...bbbbbbbbbb'" in diff
    diff = dict_diff({}, 'not a dict')
    assert diff == 'Diff between nipype inputs failed:\n* Cached inputs: {}\n* New inputs: not a dict'