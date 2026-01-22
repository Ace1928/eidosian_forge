import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_tab(self):
    self.assertEqual(b'\tbar\t', _parse_string(b'\\tbar\\t'))