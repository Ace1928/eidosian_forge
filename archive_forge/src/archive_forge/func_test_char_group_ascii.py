import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_char_group_ascii(self):
    self.assertMatchBasenameAndFullpath([('[[:ascii:]]', ['a', 'Q', '^', '.'], ['Ì', '茶']), ('[^[:ascii:]]', ['Ì', '茶'], ['a', 'Q', '^', '.'])])