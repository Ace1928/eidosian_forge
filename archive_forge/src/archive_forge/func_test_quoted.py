import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_quoted(self):
    self.assertEqual(b' foo', _parse_string(b'" foo"'))
    self.assertEqual(b'\tfoo', _parse_string(b'"\\tfoo"'))