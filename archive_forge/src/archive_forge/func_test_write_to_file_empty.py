import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_write_to_file_empty(self):
    c = ConfigFile()
    f = BytesIO()
    c.write_to_file(f)
    self.assertEqual(b'', f.getvalue())