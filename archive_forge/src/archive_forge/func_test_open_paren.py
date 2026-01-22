import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def test_open_paren(self):
    self.assertAccess('<foo(|>')