import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_empty_line_before_section(self):
    cf = self.from_file(b'\n[section]\n')
    self.assertEqual(ConfigFile({(b'section',): {}}), cf)