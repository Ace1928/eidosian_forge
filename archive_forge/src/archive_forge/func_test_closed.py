import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def test_closed(self):
    self.assertAccess('"<as|df>"')
    self.assertAccess('"<asdf|>"')
    self.assertAccess('"<|asdf>"')
    self.assertAccess("'<asdf|>'")
    self.assertAccess("'<|asdf>'")
    self.assertAccess("'''<asdf|>'''")
    self.assertAccess('"""<asdf|>"""')
    self.assertAccess('asdf.afd("a") + "<asdf|>"')