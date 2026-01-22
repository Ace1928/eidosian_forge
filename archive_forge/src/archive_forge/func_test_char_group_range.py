import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_char_group_range(self):
    self.assertMatchBasenameAndFullpath([('[a-z]', ['a', 'q', 'f'], ['A', 'Q', 'F']), ('[^a-z]', ['A', 'Q', 'F'], ['a', 'q', 'f']), ('[!a-z]foo', ['Afoo', '.foo'], ['afoo', 'ABfoo']), ('foo[!a-z]bar', ['fooAbar', 'foo.bar'], ['foojbar']), ('[ -0茶]', [' ', '$', '茶'], ['\x1f']), ('[^ -0茶]', ['\x1f'], [' ', '$', '茶'])])