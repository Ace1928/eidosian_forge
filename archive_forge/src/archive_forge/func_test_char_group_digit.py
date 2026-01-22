import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_char_group_digit(self):
    self.assertMatchBasenameAndFullpath([('[[:digit:]]', ['0', '5', '٣', '۹', '༡'], ['T', 'q', ' ', '茶', '.']), ('[^[:digit:]]', ['T', 'q', ' ', '茶', '.'], ['0', '5', '٣', '۹', '༡'])])