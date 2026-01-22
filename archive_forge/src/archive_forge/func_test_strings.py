import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def test_strings(self):
    self.assertAccess('"hey".<a|>')
    self.assertAccess('"hey"|')
    self.assertAccess('"hey"|.a')
    self.assertAccess('"hey".<a|b>')
    self.assertAccess('"hey".asdf d|')
    self.assertAccess('"hey".<|>')