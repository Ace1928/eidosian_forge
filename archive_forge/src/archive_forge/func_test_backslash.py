import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_backslash(self):
    self.assertEqual(b'foo\\\\', _escape_value(b'foo\\'))