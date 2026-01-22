import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_char_group_space(self):
    self.assertMatchBasenameAndFullpath([('[[:space:]]', [' ', '\t', '\n', '\xa0', '\u2000', '\u2002'], ['a', '-', '茶', '.']), ('[^[:space:]]', ['a', '-', '茶', '.'], [' ', '\t', '\n', '\xa0', '\u2000', '\u2002'])])