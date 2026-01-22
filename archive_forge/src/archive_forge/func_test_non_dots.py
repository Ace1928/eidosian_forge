import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def test_non_dots(self):
    self.assertAccess('].asdf|')
    self.assertAccess(').asdf|')
    self.assertAccess('foo[0].asdf|')
    self.assertAccess('foo().asdf|')
    self.assertAccess('foo().|')
    self.assertAccess('foo().asdf.|')
    self.assertAccess('foo[0].asdf.|')