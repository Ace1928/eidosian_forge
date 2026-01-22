import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_char_group_alnum(self):
    self.assertMatchBasenameAndFullpath([('[[:alnum:]]', ['a', 'Z', 'ž', '茶'], [':', '-', '●', '.']), ('[^[:alnum:]]', [':', '-', '●', '.'], ['a'])])